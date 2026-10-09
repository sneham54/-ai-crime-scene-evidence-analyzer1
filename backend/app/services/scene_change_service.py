"""
Scene Change Detection for AI Crime Scene Evidence Analyzer
Implements: image alignment (ORB + homography), SSIM, absolute difference,
changed-region highlighting and a change score (synopsis objective 6).
"""
import cv2
import numpy as np
import os
import uuid
from flask import current_app

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


class SceneChangeService:
    def compare(self, image_path_a: str, image_path_b: str, align: bool = True):
        """
        Compare two crime-scene views.
        Returns change score, SSIM, difference heatmap path, overlay path, summary.
        """
        img_a = cv2.imread(image_path_a)
        img_b = cv2.imread(image_path_b)
        if img_a is None or img_b is None:
            raise ValueError("Could not read one or both images")

        # Resize B to match A for fair comparison
        h, w = img_a.shape[:2]
        img_b = cv2.resize(img_b, (w, h))

        aligned_b = img_b
        alignment_used = False
        inliers = 0

        if align:
            aligned_b, inliers, ok = self._align_images(img_a, img_b)
            alignment_used = ok

        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(aligned_b, cv2.COLOR_BGR2GRAY)

        # Structural Similarity
        if HAS_SKIMAGE:
            ssim_score, ssim_map = ssim(gray_a, gray_b, full=True)
            ssim_score = float(ssim_score)
        else:
            # Fallback: correlation-based similarity
            ssim_score = self._correlation_similarity(gray_a, gray_b)
            ssim_map = None

        # Absolute difference
        diff = cv2.absdiff(gray_a, gray_b)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        # Morphological clean-up
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        changed_pixels = int(np.count_nonzero(thresh))
        total_pixels = thresh.size
        change_ratio = changed_pixels / total_pixels if total_pixels else 0.0

        # Change score 0–100 (higher = more change)
        # Combine low SSIM and high pixel change
        change_score = int(round(min(100, (1.0 - ssim_score) * 55 + change_ratio * 100 * 0.45)))

        # Contours of changed regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_regions = []
        overlay = img_a.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            significant_regions.append({
                "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
                "area": int(area),
            })
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

        # Colorize difference heatmap
        heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        heatmap = cv2.addWeighted(img_a, 0.45, heatmap, 0.55, 0)

        out_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], "scene_change")
        os.makedirs(out_dir, exist_ok=True)
        uid = uuid.uuid4().hex[:10]
        heatmap_path = os.path.join(out_dir, f"heatmap_{uid}.jpg")
        overlay_path = os.path.join(out_dir, f"overlay_{uid}.jpg")
        cv2.imwrite(heatmap_path, heatmap)
        cv2.imwrite(overlay_path, overlay)

        # Interpretation
        if change_score >= 55:
            interpretation = "Significant scene change detected – possible object displacement or alteration."
        elif change_score >= 25:
            interpretation = "Moderate visual differences – review highlighted regions."
        else:
            interpretation = "Minimal change between the two views."

        return {
            "status": "success",
            "ssim_score": round(ssim_score, 4),
            "change_score": change_score,
            "change_ratio": round(change_ratio, 4),
            "changed_pixels": changed_pixels,
            "significant_regions": significant_regions[:30],
            "region_count": len(significant_regions),
            "alignment_used": alignment_used,
            "alignment_inliers": inliers,
            "interpretation": interpretation,
            "heatmap_image": heatmap_path,
            "overlay_image": overlay_path,
        }

    def _align_images(self, img_a, img_b):
        """ORB feature matching + homography to align img_b to img_a."""
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(2000)
        kp1, des1 = orb.detectAndCompute(gray_a, None)
        kp2, des2 = orb.detectAndCompute(gray_b, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return img_b, 0, False

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        if len(good) < 8:
            return img_b, 0, False

        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
        if H is None:
            return img_b, 0, False
        inliers = int(mask.sum()) if mask is not None else 0
        h, w = img_a.shape[:2]
        aligned = cv2.warpPerspective(img_b, H, (w, h))
        return aligned, inliers, True

    def _correlation_similarity(self, a, b):
        a = a.astype(np.float32).ravel()
        b = b.astype(np.float32).ravel()
        a = (a - a.mean()) / (a.std() + 1e-6)
        b = (b - b.mean()) / (b.std() + 1e-6)
        return float(np.clip((a * b).mean(), -1, 1) * 0.5 + 0.5)
