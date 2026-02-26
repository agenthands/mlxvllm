package mlx

import (
	"image"
	"math"
)

const (
	MergePatchSize = 28
)

// SmartResize resizes image to fit within [minPixels, maxPixels]
// while maintaining aspect ratio and aligning to MergePatchSize
func SmartResize(img image.Image, minPixels, maxPixels int) (image.Image, error) {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	currentPixels := w * h

	// If already in range, ensure grid alignment
	if currentPixels >= minPixels && currentPixels <= maxPixels {
		return alignToGrid(img)
	}

	// Calculate scale factor
	scale := 1.0
	if currentPixels < minPixels {
		scale = math.Sqrt(float64(minPixels) / float64(currentPixels))
	} else if currentPixels > maxPixels {
		scale = math.Sqrt(float64(maxPixels) / float64(currentPixels))
	}

	newW := int(math.Round(float64(w) * scale))
	newH := int(math.Round(float64(h) * scale))

	// Align to grid size
	newW = (newW / MergePatchSize) * MergePatchSize
	newH = (newH / MergePatchSize) * MergePatchSize

	// Ensure minimum size
	if newW < MergePatchSize {
		newW = MergePatchSize
	}
	if newH < MergePatchSize {
		newH = MergePatchSize
	}

	return resizeImage(img, newW, newH)
}

// CalculateGrid returns the grid dimensions for patch processing
func CalculateGrid(w, h int) (int, int) {
	return w / MergePatchSize, h / MergePatchSize
}

func alignToGrid(img image.Image) (image.Image, error) {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	gridW, gridH := CalculateGrid(w, h)
	newW := gridW * MergePatchSize
	newH := gridH * MergePatchSize

	// Ensure minimum size after grid alignment
	if newW < MergePatchSize {
		newW = MergePatchSize
	}
	if newH < MergePatchSize {
		newH = MergePatchSize
	}

	return resizeImage(img, newW, newH)
}

func resizeImage(img image.Image, w, h int) (image.Image, error) {
	// Simple bilinear resize implementation
	// In production, use a dedicated imaging library
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	srcBounds := img.Bounds()

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			srcX := x * srcBounds.Dx() / w
			srcY := y * srcBounds.Dy() / h
			dst.Set(x, y, img.At(srcBounds.Min.X+srcX, srcBounds.Min.Y+srcY))
		}
	}

	return dst, nil
}
