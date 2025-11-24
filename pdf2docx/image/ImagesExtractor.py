'''Extract images from PDF.

Both raster images and vector graphics are considered:

* Normal images like jpeg or png could be extracted with method ``page.get_text('rawdict')`` 
  and ``Page.get_images()``. Note the process for png images with alpha channel.
* Vector graphics are actually composed of a group of paths, represented by operators like
  ``re``, ``m``, ``l`` and ``c``. They're detected by finding the contours with ``opencv``.
'''

import logging
import io
from typing import Tuple, Any
import fitz
from ..common.Collection import Collection
from ..common.share import BlockType
from ..common.algorithm import (recursive_xy_cut, inner_contours, xy_project_profile)
try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None


class ImagesExtractor:
    def __init__(self, page:fitz.Page) -> None:
        '''Extract images from PDF page.
        
        Args:
            page (fitz.Page): pdf page to extract images.
        '''
        self._page = page
    

    def clip_page_to_pixmap(self, bbox:fitz.Rect=None, zoom:float=3.0):
        '''Clip page pixmap (without text) according to ``bbox``.

        Args:
            bbox (fitz.Rect, optional): Target area to clip. Defaults to None, i.e. entire page.
                Note that ``bbox`` depends on un-rotated page CS, while clipping page is based on
                the final page.
            zoom (float, optional): Improve resolution by this rate. Defaults to 3.0.

        Returns:
            fitz.Pixmap: The extracted pixmap.
        '''        
        # hide text 
        self._hide_page_text(self._page)
        
        if bbox is None:
            clip_bbox = self._page.rect
        
        # transform to the final bbox when page is rotated
        elif self._page.rotation:
            clip_bbox = bbox * self._page.rotation_matrix
            
        else:
            clip_bbox = bbox
        
        clip_bbox = clip_bbox & self._page.rect
        
        # improve resolution
        # - https://pymupdf.readthedocs.io/en/latest/faq.html#how-to-increase-image-resolution
        # - https://github.com/pymupdf/PyMuPDF/issues/181
        matrix = fitz.Matrix(zoom, zoom)

        return self._page.get_pixmap(clip=clip_bbox, matrix=matrix) # type: fitz.Pixmap


    def clip_page_to_dict(self, bbox:fitz.Rect=None, clip_image_res_ratio:float=3.0):
        '''Clip page pixmap (without text) according to ``bbox`` and convert to source image.

        Args:
            bbox (fitz.Rect, optional): Target area to clip. Defaults to None, i.e. entire page.
            clip_image_res_ratio (float, optional): Resolution ratio of clipped bitmap. Defaults to 3.0.

        Returns:
            list: A list of image raw dict.
        '''
        pix = self.clip_page_to_pixmap(bbox=bbox, zoom=clip_image_res_ratio)
        return self._to_raw_dict(pix, bbox)


    def extract_images(self, clip_image_res_ratio:float=3.0):
        '''Extract normal images with ``Page.get_images()``.

        Args:
            clip_image_res_ratio (float, optional): Resolution ratio of clipped bitmap. Defaults to 3.0.

        Returns:
            list: A list of extracted and recovered image raw dict.
        
        .. note::
            ``Page.get_images()`` contains each image only once, which may less than the real count of images in a page.
        '''
        # pdf document
        doc = self._page.parent
        rotation = self._page.rotation

        # The final view might be formed by several images with alpha channel only, as shown in issue-123. 
        # It's still inconvenient to extract the original alpha/mask image, as a compromise, extract the 
        # equivalent image by clipping the union page region for now.
        # https://github.com/dothinking/pdf2docx/issues/123

        # step 1: collect images: [(bbox, item), ..., ]
        ic = Collection()
        for item in self._page.get_images(full=True):
            item = list(item)
            item[-1] = 0            
            
            # find all occurrences referenced to this image            
            rects = self._page.get_image_rects(item)
            unrotated_page_bbox = self._page.cropbox # note the difference to page.rect
            for bbox in rects:
                # ignore small images
                if bbox.get_area()<=4: continue

                # ignore images outside page
                if not unrotated_page_bbox.intersects(bbox): continue

                # collect images
                ic.append((bbox, item))

        # step 2: group by intersection
        fun = lambda a, b: a[0].intersects(b[0])
        groups = ic.group(fun)

        # step 3: check each group
        images = []
        for group in groups:
            # clip page with the union bbox of all intersected images
            if len(group) > 1:
                clip_bbox = fitz.Rect()
                for (bbox, item) in group: clip_bbox |= bbox
                raw_dict = self.clip_page_to_dict(clip_bbox, clip_image_res_ratio)
            
            else:
                bbox, item = group[0]

                # Regarding images consist of alpha values only, the turquoise color shown in
                # the PDF is not part of the image, but part of PDF background.
                # So, just to clip page pixmap according to the right bbox
                # https://github.com/pymupdf/PyMuPDF/issues/677

                # It's not safe to identify images with alpha values only,
                # - colorspace is None, for pymupdf <= 1.23.8
                # - colorspace is always Colorspace(CS_RGB), for pymupdf==1.23.9-15 -> issue
                # - colorspace is Colorspace(CS_), for pymupdf >= 1.23.16

                # So, use extracted image info directly.
                # image item: (xref, smask, width, height, bpc, colorspace, ...), e.g.,
                # (19, 0, 331, 369, 1, '', '', 'Im1', 'FlateDecode', 0)
                # (20, 24, 1265, 1303, 8, 'DeviceRGB', '', 'Im2', 'FlateDecode', 0)
                # (21, 0, 331, 369, 1, '', '', 'Im3', 'CCITTFaxDecode', 0)
                # (22, 25, 1265, 1303, 8, 'DeviceGray', '', 'Im4', 'DCTDecode', 0)
                # (23, 0, 1731, 1331, 8, 'DeviceGray', '', 'Im5', 'DCTDecode', 0)
                if item[5]=='':
                    raw_dict = self.clip_page_to_dict(bbox, clip_image_res_ratio)
                
                # normal images
                else:
                    # Use the same logic as our fixed pdf_image_extractor.py
                    # First analyze the transform to decide extraction method
                    xref = item[0]
                    is_simple, rotation_angle, image_ext = self._analyze_image_transform(self._page, xref, bbox)

                    # Combine page rotation and image rotation
                    total_rotation = (rotation + rotation_angle) % 360

                    if is_simple:
                        # Simple transform: extract raw data directly, then apply rotation
                        try:
                            base_image = doc.extract_image(xref)
                            image_data = base_image["image"]

                            # Apply rotation using our processing method
                            processed_image_data = self._apply_rotation_to_raw_image(
                                image_data, total_rotation
                            )

                            # Create pixmap from processed data for _to_raw_dict
                            pix = fitz.Pixmap(processed_image_data)
                            raw_dict = self._to_raw_dict(pix, bbox)
                            raw_dict['image'] = processed_image_data

                        except Exception as e:
                            logging.warning(f"Direct extraction failed, falling back to _recover_pixmap: {e}")
                            # Fallback to original method
                            pix = self._recover_pixmap(doc, item)
                            raw_dict = self._to_raw_dict(pix, bbox)
                            raw_dict['image'] = self._process_image_rotation(doc, item, pix, bbox, rotation)
                    else:
                        # Complex transform: use rendering like our code does
                        # Get original image info to determine appropriate rendering resolution
                        try:
                            base_image = doc.extract_image(xref)
                            original_img_width = base_image["width"]
                            original_img_height = base_image["height"]
                        except:
                            original_img_width = None
                            original_img_height = None

                        # Calculate scale factor to maintain original resolution
                        bbox_width = abs(bbox.x1 - bbox.x0)
                        bbox_height = abs(bbox.y1 - bbox.y0)

                        if original_img_width and original_img_height and bbox_width > 0 and bbox_height > 0:
                            scale_x = original_img_width / bbox_width
                            scale_y = original_img_height / bbox_height
                            scale = min(scale_x, scale_y)
                            scale = min(scale, 4.0)
                            scale = max(scale, 2.0)
                        else:
                            scale = 2.0

                        logging.debug(f"Rendering complex transform: scale={scale:.2f}x, rotation={total_rotation}°")

                        # Render with calculated scale factor
                        matrix = fitz.Matrix(scale, scale)
                        pixmap = self._page.get_pixmap(matrix=matrix, clip=bbox)

                        # Apply rotation if needed
                        if total_rotation != 0:
                            raw_dict = self._to_raw_dict(pixmap, bbox)
                            raw_dict['image'] = self._rotate_image(pixmap, -total_rotation)
                        else:
                            raw_dict = self._to_raw_dict(pixmap, bbox)

            images.append(raw_dict)

        return images    
        
    
    def detect_svg_contours(self, min_svg_gap_dx:float, min_svg_gap_dy:float, min_w:float, min_h:float):
        '''Find contour of potential vector graphics.

        Args:
            min_svg_gap_dx (float): Merge svg if the horizontal gap is less than this value.
            min_svg_gap_dy (float): Merge svg if the vertical gap is less than this value.
            min_w (float): Ignore contours if the bbox width is less than this value.
            min_h (float): Ignore contours if the bbox height is less than this value.

        Returns:
            list: A list of potential svg region: (external_bbox, inner_bboxes:list).
        '''
        import cv2 as cv

        # clip page and convert to opencv image
        pixmap = self.clip_page_to_pixmap(zoom=1.0)
        src = self._pixmap_to_cv_image(pixmap)

        # gray and binary
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(gray, 253, 255, cv.THRESH_BINARY_INV)
        
        # external bbox: split images with recursive xy cut
        external_bboxes = recursive_xy_cut(binary, min_dx=min_svg_gap_dx, min_dy=min_svg_gap_dy)        
        
        # inner contours
        grouped_inner_bboxes = [inner_contours(binary, bbox, min_w, min_h) for bbox in external_bboxes]

        # combined external and inner contours
        groups = list(zip(external_bboxes, grouped_inner_bboxes))
            
        
        # plot detected images for debug
        debug = False
        if debug:
            # plot projection profile for each sub-image
            for i, (x0, y0, x1, y1) in enumerate(external_bboxes):
                arr = xy_project_profile(src[y0:y1, x0:x1, :], binary[y0:y1, x0:x1])
                cv.imshow(f'sub-image-{i}', arr)

            for bbox, inner_bboxes in groups:
                # plot external bbox
                x0, y0, x1, y1 = bbox
                cv.rectangle(src, (x0, y0), (x1, y1), (255,0,0), 1)

                # plot inner bbox
                for u0, v0, u1, v1 in inner_bboxes:
                    cv.rectangle(src, (u0, v0), (u1, v1), (0,0,255), 1)

            cv.imshow("img", src)
            cv.waitKey(0)

        return groups


    @staticmethod
    def _to_raw_dict(image:fitz.Pixmap, bbox:fitz.Rect):
        '''Store Pixmap ``image`` to raw dict.

        Args:
            image (fitz.Pixmap): Pixmap to store.
            bbox (fitz.Rect): Boundary box the pixmap.

        Returns:
            dict: Raw dict of the pixmap.
        '''
        return {
            'type': BlockType.IMAGE.value,
            'bbox': tuple(bbox),
            'width': image.width,
            'height': image.height,
            'image': image.tobytes()
        }


    @staticmethod
    def _rotate_image(pixmap:fitz.Pixmap, rotation:int):
        '''Rotate image represented by image bytes.

        Args:
            pixmap (fitz.Pixmap): Image to rotate.
            rotation (int): Rotation angle.
        
        Return: image bytes.
        '''
        import cv2 as cv
        import numpy as np

        # convert to opencv image
        img = ImagesExtractor._pixmap_to_cv_image(pixmap)
        h, w = img.shape[:2] # get image height, width

        # calculate the center of the image
        x0, y0 = w//2, h//2

        # default scale value for now -> might be extracted from PDF page property    
        scale = 1.0

        # rotation matrix
        matrix = cv.getRotationMatrix2D((x0, y0), rotation, scale)

        # calculate the final dimension
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
    
        # compute the new bounding dimensions of the image
        W = int((h * sin) + (w * cos))
        H = int((h * cos) + (w * sin))
    
        # adjust the rotation matrix to take into account translation
        matrix[0, 2] += (W / 2) - x0
        matrix[1, 2] += (H / 2) - y0
        
        # perform the rotation holding at the center        
        rotated_img = cv.warpAffine(img, matrix, (W, H))

        # convert back to bytes
        _, im_png = cv.imencode('.png', rotated_img)
        return im_png.tobytes()


    @staticmethod
    def _hide_page_text(page:fitz.Page):
        '''Hide page text before clipping page.'''
        # NOTE: text might exist in both content stream and form object stream
        # - content stream, i.e. direct page content
        # - form object, i.e. contents referenced by this page
        xref_list = [xref for (xref, name, invoker, bbox) in page.get_xobjects()]
        xref_list.extend(page.get_contents())        

        # render Tr: set the text rendering mode
        # - 3: neither fill nor stroke the text -> invisible
        # read more:
        # - https://github.com/pymupdf/PyMuPDF/issues/257
        # - https://www.adobe.com/content/dam/acom/en/devnet/pdf/pdfs/pdf_reference_archives/PDFReference.pdf
        doc = page.parent # type: fitz.Document
        for xref in xref_list:
            stream = doc.xref_stream(xref).replace(b'BT', b'BT 3 Tr') \
                                             .replace(b'Tm', b'Tm 3 Tr') \
                                             .replace(b'Td', b'Td 3 Tr')
            doc.update_stream(xref, stream)
   
    @staticmethod
    def _recover_pixmap(doc:fitz.Document, item:list):
        """Restore pixmap with soft mask considered.
        
        References:

            * https://pymupdf.readthedocs.io/en/latest/document.html#Document.getPageImageList        
            * https://pymupdf.readthedocs.io/en/latest/faq.html#how-to-handle-stencil-masks
            * https://github.com/pymupdf/PyMuPDF/issues/670

        Args:
            doc (fitz.Document): pdf document.
            item (list): image instance of ``page.get_images()``.

        Returns:
            fitz.Pixmap: Recovered pixmap with soft mask considered.
        """
        # data structure of `item`:
        # (xref, smask, width, height, bpc, colorspace, ...)
        x = item[0]  # xref of PDF image
        s = item[1]  # xref of its /SMask

        # base image
        pix = fitz.Pixmap(doc, x)

        # reconstruct the alpha channel with the smask if exists
        if s > 0:
            mask = fitz.Pixmap(doc, s)
            if pix.alpha:
                temp = fitz.Pixmap(pix, 0)  # make temp pixmap w/o the alpha
                pix = None  # release storage
                pix = temp
            
            # check dimension
            if pix.width==mask.width and pix.height==mask.height:
                pix = fitz.Pixmap(pix, mask)  # now compose final pixmap
            else:
                logging.warning('Ignore image due to inconsistent size of color and mask pixmaps: %s', item)

        # we may need to adjust something for CMYK pixmaps here -> 
        # recreate pixmap in RGB color space if necessary
        # NOTE: pix.colorspace may be None for images with alpha channel values only
        if 'CMYK' in item[5].upper():
            pix = fitz.Pixmap(fitz.csRGB, pix)

        return pix


    def _apply_rotation_to_raw_image(self, image_data:bytes, rotation:int) -> bytes:
        """
        Apply rotation to raw image data using PIL, exactly like our working code.

        Args:
            image_data: Raw image bytes
            rotation: Rotation angle (counter-clockwise, 0/90/180/270)

        Returns:
            bytes: Rotated image data as bytes
        """
        if Image is None or ImageOps is None or rotation == 0:
            return image_data

        try:
            # Open image with PIL
            image = Image.open(io.BytesIO(image_data))

            # Step 1: Apply EXIF orientation correction first (like our working code)
            image = ImageOps.exif_transpose(image)

            # Step 2: Apply PDF rotation
            if rotation != 0:
                image = image.rotate(
                    rotation,
                    expand=True,
                    resample=Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
                )

            # Convert back to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            return img_byte_arr.getvalue()

        except Exception as e:
            logging.warning(f"Failed to apply rotation to raw image: {e}")
            return image_data


    def _process_image_rotation(self, doc:fitz.Document, item:list, pixmap:fitz.Pixmap, bbox:fitz.Rect, page_rotation:int) -> bytes:
        """
        Process image rotation by analyzing the transform matrix and applying correct rotation.

        This method implements the same logic as our fixed pdf_image_extractor.py:
        1. Analyze the image transform matrix to detect rotation
        2. Apply EXIF orientation correction first
        3. Apply PDF rotation transformation

        Args:
            doc (fitz.Document): PDF document
            item (list): Image item from get_images()
            pixmap (fitz.Pixmap): The recovered pixmap (already processed by _recover_pixmap)
            bbox (fitz.Rect): Bounding box of the image
            page_rotation (int): Page rotation angle

        Returns:
            bytes: Processed image data as bytes
        """
        try:
            # Get image xref
            xref = item[0]

            # Skip processing if PIL is not available
            if Image is None or ImageOps is None:
                logging.debug("PIL not available, using original rotation method")
                if page_rotation:
                    return self._rotate_image(pixmap, -page_rotation)
                else:
                    return pixmap.tobytes()

            # Analyze transform matrix
            _, rotation_angle, _ = self._analyze_image_transform(self._page, xref, bbox)

            # Combine page rotation and image rotation
            total_rotation = (page_rotation + rotation_angle) % 360

            logging.debug(f"Image rotation: page={page_rotation}°, transform={rotation_angle}°, total={total_rotation}°")

            # Convert the already-processed pixmap to PIL Image
            # IMPORTANT: Use pixmap.tobytes() NOT doc.extract_image() to preserve _recover_pixmap() processing
            image_data = pixmap.tobytes("png")  # Get PNG bytes from the processed pixmap
            image = Image.open(io.BytesIO(image_data))

            # Step 1: Apply EXIF orientation correction first
            # This fixes camera orientation issues before applying PDF transforms
            try:
                image = ImageOps.exif_transpose(image)
                logging.debug("Applied EXIF orientation correction")
            except Exception as e:
                logging.debug(f"EXIF correction failed: {e}")

            # Step 2: Apply PDF rotation if needed
            if total_rotation != 0:
                logging.debug(f"Applying PDF rotation: {total_rotation}°")
                # PIL's rotate() uses counter-clockwise rotation
                # expand=True ensures the image is not cropped
                image = image.rotate(
                    total_rotation,
                    expand=True,
                    resample=Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
                )

            # Convert back to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            return img_byte_arr.getvalue()

        except Exception as e:
            logging.warning(f"Failed to process image rotation with new method, falling back to original: {e}")
            # Fallback to original method
            if page_rotation:
                return self._rotate_image(pixmap, -page_rotation)
            else:
                return pixmap.tobytes()


    def _analyze_image_transform(self, page:fitz.Page, xref:int, rect_info:Any) -> Tuple[bool, int, str]:
        """
        Analyze image transform matrix to detect rotation and determine if rendering is needed.

        This method implements the same logic as our fixed pdf_image_extractor.py:
        - Supports all rotation angles (0°, 90°, 180°, 270°)
        - Detects flip transforms
        - Handles complex transformations

        Args:
            page (fitz.Page): PDF page
            xref (int): Image xref
            rect_info (Any): Rectangle information

        Returns:
            Tuple[bool, int, str]: (is_simple_transform, rotation_angle, image_format)
        """
        try:
            # Get original image info
            base_image = page.parent.extract_image(xref)
            image_ext = base_image.get("ext", "png")

            # Try to get transform information
            try:
                image_instances = page.get_image_bbox(xref)
            except (ValueError, RuntimeError) as e:
                # "bad image name" or other PyMuPDF internal errors
                # This happens with inline images or special image objects
                logging.debug(f"Cannot get image transform info (xref={xref}): {e}, using rendering mode")
                return False, 0, image_ext

            if not image_instances:
                # No transform info, use original image
                return True, 0, image_ext

            # Use the first instance's transform matrix
            # get_image_bbox returns a list of (bbox, matrix) tuples
            bbox, transform = image_instances[0]

            # Helper function to check if float is close to target value
            def is_near(value: float, target: float, tol: float = 0.01) -> bool:
                return abs(value - target) < tol

            # Extract transform matrix elements
            # Transform matrix format: [a, b, c, d, e, f]
            # a, d: scale factors
            # b, c: rotation/shear factors
            # e, f: translation
            a, b, c, d = transform.a, transform.b, transform.c, transform.d

            # Detect rotation angle
            rotation_angle = 0
            is_simple = True

            # Check rotation patterns
            # 0°: a≈1, d≈1, b≈0, c≈0
            if is_near(a, 1) and is_near(d, 1) and is_near(b, 0) and is_near(c, 0):
                rotation_angle = 0
                is_simple = True

            # 90° clockwise (or 270° counter-clockwise): a≈0, d≈0, b≈1, c≈-1
            elif is_near(a, 0) and is_near(d, 0) and is_near(b, 1) and is_near(c, -1):
                rotation_angle = 270  # PIL uses counter-clockwise, so 90° clockwise = 270° counter-clockwise
                is_simple = True

            # 90° counter-clockwise (or 270° clockwise): a≈0, d≈0, b≈-1, c≈1
            elif is_near(a, 0) and is_near(d, 0) and is_near(b, -1) and is_near(c, 1):
                rotation_angle = 90
                is_simple = True

            # 180°: a≈-1, d≈-1, b≈0, c≈0
            elif is_near(a, -1) and is_near(d, -1) and is_near(b, 0) and is_near(c, 0):
                rotation_angle = 180
                is_simple = True

            # Horizontal flip: a≈-1, d≈1, b≈0, c≈0
            elif is_near(a, -1) and is_near(d, 1) and is_near(b, 0) and is_near(c, 0):
                # Horizontal flip requires rendering
                is_simple = False

            # Vertical flip: a≈1, d≈-1, b≈0, c≈0
            elif is_near(a, 1) and is_near(d, -1) and is_near(b, 0) and is_near(c, 0):
                # Vertical flip requires rendering
                is_simple = False

            else:
                # Complex transformation (scaling, shearing, etc.), requires rendering
                is_simple = False
                logging.debug(
                    f"Detected complex transform: a={a:.3f}, b={b:.3f}, c={c:.3f}, d={d:.3f}"
                )

            return is_simple, rotation_angle, image_ext

        except Exception as e:
            # Catch other unexpected exceptions
            logging.warning(f"Unexpected error analyzing transform matrix, falling back to rendering mode: {e}")
            return False, 0, "png"


    @staticmethod
    def _pixmap_to_cv_image(pixmap:fitz.Pixmap):
        '''Convert fitz Pixmap to opencv image.

        Args:
            pixmap (fitz.Pixmap): PyMuPDF Pixmap.
        '''
        import cv2 as cv
        import numpy as np
        img_byte = pixmap.tobytes()
        return cv.imdecode(np.frombuffer(img_byte, np.uint8), cv.IMREAD_COLOR)
