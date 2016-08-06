package matrix

import (
	"math/rand"
	"reflect"
	"testing"
	"time"
)

var s1 = [][]float32{
	[]float32{2, 3},
	[]float32{4, 5},
}

var s2 = [][]float32{
	[]float32{10, 20},
	[]float32{30, 40},
}

var s3 = [][]float32{
	[]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
	[]float32{11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
	[]float32{21, 22, 23, 24, 25, 26, 27, 28, 29, 30},
	[]float32{41, 42, 43, 44, 45, 46, 47, 48, 49, 50},
}

var one = NewMatrix([][]float32{
	[]float32{1, 1, 1},
	[]float32{1, 1, 1},
	[]float32{1, 1, 1},
})

func GetMatrix(rows, cols int) *Matrix {
	rand.Seed(time.Now().UnixNano())
	s := make([][]float32, rows)
	for y := 0; y < rows; y++ {
		col := make([]float32, cols)
		for x := 0; x < cols; x++ {
			col[x] = rand.Float32()
		}
		s[y] = col
	}
	return NewMatrix(s)
}

func TestNewMatrix(t *testing.T) {
	NewMatrix([][]float32{
		{1},
		{1},
		{1},
		{1},
		{1},
	})
	NewMatrix([][]float32{
		{1, 1, 1, 1, 1},
	})
	NewMatrix([][]float32{
		{1, 1, 1, 1},
		{1, 1, 1, 1},
		{1, 1, 1, 1},
	})
}

func TestEquals(t *testing.T) {
	m1 := NewMatrix(s1)
	m2 := NewMatrix(s1)
	if !m1.Equals(m2) {
		t.Error("not same")
	}
	m2 = NewMatrix(s2)
	if m1.Equals(m2) {
		t.Error("same")
	}
}

func TestAdd(t *testing.T) {
	m1 := NewMatrix(s1)
	m2 := NewMatrix(s2)
	_, err := Add(m1, m2)
	if err != nil {
		t.Fatal(err)
	}
}

func TestSub(t *testing.T) {
	m1 := NewMatrix(s1)
	m2 := NewMatrix(s2)
	_, err := Sub(m1, m2)
	if err != nil {
		t.Fatal(err)
	}
}

func TestMul(t *testing.T) {
	m1 := NewMatrix(s1)
	m2 := NewMatrix(s2)
	_, err := Mul(m1, m2)
	if err != nil {
		t.Fatal(err)
	}
}

func TestElemAdd(t *testing.T) {
	m1 := NewMatrix(s1)
	m2 := NewMatrix(s2)
	ans := NewMatrix([][]float32{
		{12, 23},
		{34, 45},
	})

	res, err := ElemAdd(m1, m2)
	if err != nil {
		t.Error(err)
	}
	if !res.Equals(ans) {
		t.Error("not same")
		ans.Show()
		res.Show()
	}
}

func TestElemSub(t *testing.T) {
	m1 := NewMatrix(s1)
	m2 := NewMatrix(s2)
	ans := NewMatrix([][]float32{
		{-8.000000, -17.000000},
		{-26.000000, -35.000000},
	})

	res, err := ElemSub(m1, m2)
	if err != nil {
		t.Error(err)
	}
	if !res.Equals(ans) {
		t.Error("not same")
		ans.Show()
		res.Show()
	}
}

func TestElemMul(t *testing.T) {
	m1 := NewMatrix(s1)
	m2 := NewMatrix(s2)
	ans := NewMatrix([][]float32{
		{20, 60},
		{120, 200},
	})

	res, err := ElemMul(m1, m2)
	if err != nil {
		t.Error(err)
	}
	if !res.Equals(ans) {
		t.Error("not same")
		ans.Show()
		res.Show()
	}
}

func TestElemDiv(t *testing.T) {
	m1 := NewMatrix([][]float32{
		{10, 20},
		{40, 60},
	})
	m2 := NewMatrix([][]float32{
		{2, 4},
		{4, 6},
	})
	ans := NewMatrix([][]float32{
		{5, 5},
		{10, 10},
	})

	res, err := ElemDiv(m1, m2)
	if err != nil {
		t.Error(err)
	}
	if !res.Equals(ans) {
		t.Error("not same")
		ans.Show()
		res.Show()
	}
}

func TestBroadcastFunc(t *testing.T) {
	m := NewMatrix([][]float32{
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
	})
	ans := NewMatrix([][]float32{
		{2, 2, 2},
		{2, 2, 2},
		{2, 2, 2},
	})
	addOne := m.BroadcastFunc(func(e float32, a ...interface{}) float32 {
		return e + a[0].(float32)
	}, float32(1.0))

	if !addOne.Equals(ans) {
		t.Error("Not Same")
		ans.Show()
		addOne.Show()
	}
}

func BenchmarkBroadcastFunc(b *testing.B) {
	for i := 0; i < b.N; i++ {
		m := GetMatrix(480, 360)
		m.BroadcastFunc(func(e float32, a ...interface{}) float32 {
			return e + a[0].(float32)
		}, float32(10.0))
	}
}

func TestBroadcastAdd(t *testing.T) {
	ans := NewMatrix([][]float32{
		[]float32{2, 2, 2},
		[]float32{2, 2, 2},
		[]float32{2, 2, 2},
	})
	res := one.BroadcastAdd(1)
	if !ans.Equals(res) {
		t.Errorf("Not same")
		res.Show()
	}
}

func TestBroadcastSub(t *testing.T) {
	ans := NewMatrix([][]float32{
		[]float32{0, 0, 0},
		[]float32{0, 0, 0},
		[]float32{0, 0, 0},
	})
	res := one.BroadcastSub(1)
	if !ans.Equals(res) {
		t.Errorf("Not same.")
		res.Show()
	}
}

func TestBroadcastMul(t *testing.T) {
	ans := NewMatrix([][]float32{
		[]float32{2, 2, 2},
		[]float32{2, 2, 2},
		[]float32{2, 2, 2},
	})
	res := one.BroadcastMul(2)
	if !ans.Equals(res) {
		t.Errorf("Not same.")
		res.Show()
	}
}

func TestBoradcastDiv(t *testing.T) {
	ans := NewMatrix([][]float32{
		[]float32{1, 1, 1},
		[]float32{1, 1, 1},
		[]float32{1, 1, 1},
	})
	res := one.BroadcastDiv(1)
	if !ans.Equals(res) {
		t.Errorf("Not same.")
		res.Show()
	}
}

func TestZeroPad(t *testing.T) {
	m1 := NewMatrix(s1)
	m1.Pad(7, Zero)
}

func TestEdgePad(t *testing.T) {
	m1 := NewMatrix(s1)
	m1.Pad(7, Edge)
}

func BenchmarkEdge(b *testing.B) {
	for i := 0; i < b.N; i++ {
		m := GetMatrix(480, 360)
		err := m.Pad(7, Edge)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func TestSlice2d(t *testing.T) {
	m := [][]float32{
		{1, 1, 1, 0, 0},
		{0, 1, 1, 1, 0},
		{0, 0, 1, 1, 1},
		{0, 0, 1, 1, 0},
		{0, 1, 1, 0, 0},
	}
	ans := [][]float32{
		{1, 1, 1},
		{0, 1, 1},
		{0, 0, 1},
	}
	res := Slice2d(m, 0, 3, 0, 3)
	if !reflect.DeepEqual(res, ans) {
		t.Error("Not same")
		t.Log(res)
		t.Log(ans)
	}

}

func TestConvolve2d(t *testing.T) {
	m := NewMatrix([][]float32{
		{1, 1, 1, 0, 0},
		{0, 1, 1, 1, 0},
		{0, 0, 1, 1, 1},
		{0, 0, 1, 1, 0},
		{0, 1, 1, 0, 0},
	})
	f := NewMatrix([][]float32{
		{1, 0, 1},
		{0, 1, 0},
		{1, 0, 1},
	})
	ans := NewMatrix([][]float32{
		{4, 3, 4},
		{2, 4, 3},
		{2, 3, 4},
	})
	res, err := m.Convolve2d(f, 1, 0, Max)
	if err != nil {
		t.Fatal(err)
	}

	if same := res.Equals(ans); !same {
		t.Error("not same")
		ans.Show()
		res.Show()
	}
}

func TestConvolve2dStride(t *testing.T) {
	m := NewMatrix([][]float32{
		{0, 2, 0, 0, 1},
		{1, 2, 0, 0, 1},
		{2, 2, 1, 2, 2},
		{0, 0, 1, 2, 1},
		{2, 1, 1, 1, 0},
	})
	f1 := NewMatrix([][]float32{
		{-1, -1, -1},
		{1, 0, 0},
		{0, 0, 1},
	})
	f2 := NewMatrix([][]float32{
		{0, 0, 1},
		{1, 1, 0},
		{1, 0, 0},
	})
	a := NewMatrix([][]float32{
		{2, 5, 4},
		{-2, 7, 4},
		{3, 1, -3},
	})
	m, err := m.Convolve2d(f1, 2, 1, Max)
	if err != nil {
		t.Error(err)
	}

	m, err = m.Convolve2d(f2, 2, 0, Max)
	if err != nil {
		t.Error(err)
	}
	if same := m.Equals(a); !same {
		t.Error("not same")
		a.Show()
		m.Show()
	}

}

func TestPooling(t *testing.T) {
	m := NewMatrix([][]float32{
		{12, 20, 30, 0},
		{8, 12, 2, 0},
		{34, 70, 37, 4},
		{112, 100, 25, 12},
	})
	ans1 := NewMatrix([][]float32{
		{20, 30},
		{112, 37},
	})

	ans2 := NewMatrix([][]float32{
		{13, 8},
		{79, 19.5},
	})

	t.Log("Test for max pooling.")
	res1 := m.Pooling(2, 2, Max)

	if !res1.Equals(ans1) {
		t.Error("Not same")
		ans1.Show()
		res1.Show()
	}

	t.Log("Test for average pooling.")
	res2 := m.Pooling(2, 2, Avg)

	if !res2.Equals(ans2) {
		t.Error("Not same")
		ans2.Show()
		res2.Show()
	}
}

func TestT(t *testing.T) {
	m1 := NewMatrix([][]float32{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	})
	ans1 := NewMatrix([][]float32{
		{1, 4, 7},
		{2, 5, 8},
		{3, 6, 9},
	})

	res1 := m1.T()

	if !ans1.Equals(res1) {
		t.Error("Not same")
	}

	m2 := NewMatrix([][]float32{
		{1, 2},
		{3, 4},
		{5, 6},
	})

	ans2 := NewMatrix([][]float32{
		{1, 3, 5},
		{2, 4, 6},
	})

	res2 := m2.T()

	if !ans2.Equals(res2) {
		t.Error("Not same")
	}
}

func TestFlatten(t *testing.T) {
	m1 := NewMatrix([][]float32{
		{1, 1, 1, 1},
		{1, 1, 1, 1},
		{1, 1, 1, 1},
	})

	ans1 := []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}

	f1 := m1.Flatten()

	if !reflect.DeepEqual(f1, ans1) {
		t.Error("Not same")
		t.Log(ans1)
		t.Log(f1)
	}

	m2 := NewMatrix([][]float32{
		{1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1},
		{2, 2, 2, 2, 2, 2},
	})

	ans2 := []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2}

	f2 := m2.Flatten()

	if !reflect.DeepEqual(f2, ans2) {
		t.Error("Not same")
		t.Log(ans2)
		t.Log(f2)
	}
}

func TestReshape(t *testing.T) {
	m := NewMatrix([][]float32{
		{0, 0, 0},
		{0, 0, 0},
		{0, 0, 0},
		{0, 0, 0},
	})

	ans := NewMatrix([][]float32{
		{0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0},
	})

	res, err := m.Reshape(2, 6)
	if err != nil {
		t.Error(err)
	}

	if !res.Equals(ans) {
		t.Errorf("Not same.")
	}
}

func TestClip(t *testing.T) {
	m := NewMatrix([][]float32{
		{-1, 0, 123, -123},
		{32, 0, 0.1, 0.2},
		{2, 0, -0.3, 0.003},
	})

	ans := NewMatrix([][]float32{
		{0.0, 0.0, 1.0, 0.0},
		{1.0, 0.0, 0.1, 0.2},
		{1.0, 0.0, 0.0, 0.003},
	})

	res := m.Clip(0, 1)
	if !res.Equals(ans) {
		t.Error("Not same.")
		ans.Show()
		res.Show()
	}
}

func TestIm2Col(t *testing.T) {
	m := NewMatrix([][]float32{
		{-1, -1, -1},
		{1, 0, 0},
		{0, 0, 1},
	})
	res := m.Im2Col(3, 1)
	a := NewMatrix([][]float32{
		{-1, -1, -1, 1, 0, 0, 0, 0, 1},
	})

	if !res.Equals(a) {
		t.Error("not same")
		a.Show()
		res.Show()
	}
}

func TestIm2ColWithStride(t *testing.T) {
	m := NewMatrix([][]float32{
		{0, 0, 0, 0, 0, 0, 0},
		{0, 0, 2, 0, 0, 1, 0},
		{0, 1, 2, 0, 0, 1, 0},
		{0, 2, 2, 1, 2, 2, 0},
		{0, 0, 0, 1, 2, 1, 0},
		{0, 2, 1, 1, 1, 0, 0},
		{0, 0, 0, 0, 0, 0, 0},
	})
	res := m.Im2Col(3, 2)

	a := NewMatrix([][]float32{
		{0, 0, 0, 0, 0, 2, 0, 1, 2},
		{0, 0, 0, 2, 0, 0, 2, 0, 0},
		{0, 0, 0, 0, 1, 0, 0, 1, 0},
		{0, 1, 2, 0, 2, 2, 0, 0, 0},
		{2, 0, 0, 2, 1, 2, 0, 1, 2},
		{0, 1, 0, 2, 2, 0, 2, 1, 0},
		{0, 0, 0, 0, 2, 1, 0, 0, 0},
		{0, 1, 2, 1, 1, 1, 0, 0, 0},
		{2, 1, 0, 1, 0, 0, 0, 0, 0},
	})

	if !res.Equals(a) {
		t.Error("not same")
		res.Show()
		a.Show()
	}
}
