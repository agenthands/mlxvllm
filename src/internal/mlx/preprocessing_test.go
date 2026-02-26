package mlx

import (
	"image"
	"image/color"
	"testing"
)

func TestSmartResize(t *testing.T) {
	// Create 100x100 test image
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	img.Set(50, 50, color.RGBA{255, 0, 0, 255})

	resized, err := SmartResize(img, 3136, 5720064)
	if err != nil {
		t.Fatalf("SmartResize failed: %v", err)
	}

	// Check dimensions satisfy constraints
	pixels := resized.Bounds().Dx() * resized.Bounds().Dy()
	if pixels < 3136 {
		t.Errorf("Pixels %d below min 3136", pixels)
	}
	if pixels > 5720064 {
		t.Errorf("Pixels %d above max 5720064", pixels)
	}
}

func TestCalculateGrid(t *testing.T) {
	tests := []struct {
		w, h     int
		expectGW int
		expectGH int
	}{
		{112, 224, 4, 8},   // 112/28=4, 224/28=8
		{224, 224, 8, 8},   // 224/28=8
		{56, 56, 2, 2},     // 56/28=2
	}

	for _, tt := range tests {
		gw, gh := CalculateGrid(tt.w, tt.h)
		if gw != tt.expectGW || gh != tt.expectGH {
			t.Errorf("CalculateGrid(%d,%d) = (%d,%d), want (%d,%d)",
				tt.w, tt.h, gw, gh, tt.expectGW, tt.expectGH)
		}
	}
}
