package mlx

import (
	"image"
	"image/color"
	"testing"
)

func TestSmartResize(t *testing.T) {
	tests := []struct {
		name         string
		w, h         int
		minPixels    int
		maxPixels    int
		expectMinW   int
		expectMinH   int
		expectMaxW   int
		expectMaxH   int
		expectGridOK bool
	}{
		{
			name:         "100x100 upscale to min",
			w:            100,
			h:            100,
			minPixels:    3136,
			maxPixels:    5720064,
			expectMinW:   28,
			expectMinH:   28,
			expectMaxW:   10000,
			expectMaxH:   10000,
			expectGridOK: true,
		},
		{
			name:         "image already in range - grid aligned",
			w:            224,
			h:            224,
			minPixels:    3136,
			maxPixels:    5720064,
			expectMinW:   224,
			expectMinH:   224,
			expectMaxW:   224,
			expectMaxH:   224,
			expectGridOK: true,
		},
		{
			name:         "image in range but needs grid alignment",
			w:            100,
			h:            100,
			minPixels:    5000,
			maxPixels:    5720064,
			expectMinW:   28,
			expectMinH:   28,
			expectMaxW:   5000,
			expectMaxH:   5000,
			expectGridOK: true,
		},
		{
			name:         "large image downscale",
			w:            4000,
			h:            4000,
			minPixels:    3136,
			maxPixels:    5720064,
			expectMinW:   28,
			expectMinH:   28,
			expectMaxW:   2394,
			expectMaxH:   2394,
			expectGridOK: true,
		},
		{
			name:         "very small image - minimum size",
			w:            10,
			h:            10,
			minPixels:    3136,
			maxPixels:    5720064,
			expectMinW:   28,
			expectMinH:   28,
			expectMaxW:   1000,
			expectMaxH:   1000,
			expectGridOK: true,
		},
		{
			name:         "wide image aspect ratio preserved",
			w:            2000,
			h:            500,
			minPixels:    3136,
			maxPixels:    5720064,
			expectMinW:   28,
			expectMinH:   28,
			expectMaxW:   5000,
			expectMaxH:   5000,
			expectGridOK: true,
		},
		{
			name:         "tall image aspect ratio preserved",
			w:            500,
			h:            2000,
			minPixels:    3136,
			maxPixels:    5720064,
			expectMinW:   28,
			expectMinH:   28,
			expectMaxW:   5000,
			expectMaxH:   5000,
			expectGridOK: true,
		},
		{
			name:         "exactly at min pixels",
			w:            56,
			h:            56,
			minPixels:    3136,
			maxPixels:    5720064,
			expectMinW:   56,
			expectMinH:   56,
			expectMaxW:   56,
			expectMaxH:   56,
			expectGridOK: true,
		},
		{
			name:         "28 pixel minimum enforced",
			w:            15,
			h:            15,
			minPixels:    100,
			maxPixels:    5720064,
			expectMinW:   28,
			expectMinH:   28,
			expectMaxW:   100,
			expectMaxH:   100,
			expectGridOK: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			img := image.NewRGBA(image.Rect(0, 0, tt.w, tt.h))
			img.Set(tt.w/2, tt.h/2, color.RGBA{255, 0, 0, 255})

			resized, err := SmartResize(img, tt.minPixels, tt.maxPixels)
			if err != nil {
				t.Fatalf("SmartResize failed: %v", err)
			}

			newW := resized.Bounds().Dx()
			newH := resized.Bounds().Dy()

			if newW < tt.expectMinW || newH < tt.expectMinH {
				t.Errorf("Dimensions (%d,%d) below minimum (%d,%d)", newW, newH, tt.expectMinW, tt.expectMinH)
			}
			if newW > tt.expectMaxW || newH > tt.expectMaxH {
				t.Errorf("Dimensions (%d,%d) above maximum (%d,%d)", newW, newH, tt.expectMaxW, tt.expectMaxH)
			}

			pixels := newW * newH
			if pixels < tt.minPixels {
				t.Errorf("Pixels %d below min %d", pixels, tt.minPixels)
			}
			if pixels > tt.maxPixels {
				t.Errorf("Pixels %d above max %d", pixels, tt.maxPixels)
			}

			if tt.expectGridOK {
				if newW%MergePatchSize != 0 || newH%MergePatchSize != 0 {
					t.Errorf("Dimensions (%d,%d) not aligned to grid size %d", newW, newH, MergePatchSize)
				}
			}

			gw, gh := CalculateGrid(newW, newH)
			if gw*MergePatchSize != newW || gh*MergePatchSize != newH {
				t.Errorf("Grid mismatch: grid=(%d,%d) but size=(%d,%d)", gw, gh, newW, newH)
			}
		})
	}
}

func TestCalculateGrid(t *testing.T) {
	tests := []struct {
		name     string
		w, h     int
		expectGW int
		expectGH int
	}{
		{"112x224", 112, 224, 4, 8},   // 112/28=4, 224/28=8
		{"224x224", 224, 224, 8, 8},   // 224/28=8
		{"56x56", 56, 56, 2, 2},       // 56/28=2
		{"28x28", 28, 28, 1, 1},       // Minimum
		{"1000x500", 1000, 500, 35, 17}, // 1000/28=35, 500/28=17
		{"odd dimensions", 99, 57, 3, 2}, // Integer division
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gw, gh := CalculateGrid(tt.w, tt.h)
			if gw != tt.expectGW || gh != tt.expectGH {
				t.Errorf("CalculateGrid(%d,%d) = (%d,%d), want (%d,%d)",
					tt.w, tt.h, gw, gh, tt.expectGW, tt.expectGH)
			}
		})
	}
}

func TestMergePatchSize(t *testing.T) {
	if MergePatchSize != 28 {
		t.Errorf("Expected MergePatchSize to be 28, got %d", MergePatchSize)
	}
}
