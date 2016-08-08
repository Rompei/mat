package mat

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// Matrix is object for matrix.
type Matrix struct {
	Rows uint
	Cols uint
	M    [][]float32
}

// CalcMode is flag for calculation.
type CalcMode int

const (
	// AddOP is flag for add.
	AddOP = iota + 1
	// SubOP is flag for sub.
	SubOP
)

// PadMode is flag for padding mode.
type PadMode int

const (
	// Zero is zero padding flag.
	Zero = iota + 1
	// Edge is edge padding flag.
	Edge
)

// PoolingMode is flag for pooling.
type PoolingMode int

const (
	// Max pooling
	Max = iota + 1
	// Avg shows average pooling.
	Avg
)

// NewMatrix is constructor of Matrix.
func NewMatrix(m [][]float32) *Matrix {
	return &Matrix{
		M:    m,
		Rows: uint(len(m)),
		Cols: uint(len(m[0])),
	}
}

// Zeros makes zero value matrix.
func Zeros(r, c uint) *Matrix {
	res := make([][]float32, r)
	for y := 0; y < int(r); y++ {
		res[y] = make([]float32, c)
	}
	return NewMatrix(res)
}

// Random returns random matrix.
func Random(r, c uint) *Matrix {
	rand.Seed(time.Now().UnixNano())
	res := make([][]float32, r)
	for y := 0; y < int(r); y++ {
		res[y] = make([]float32, c)
		for x := 0; x < int(c); x++ {
			res[y][x] = rand.Float32()
		}
	}
	return NewMatrix(res)
}

// GetSize returns size of the matrix.
func (m *Matrix) GetSize() (uint, uint) {
	return m.Rows, m.Cols
}

// GetRows returns the number of rows as integer.
func (m *Matrix) GetRows() int {
	return int(m.Rows)
}

// GetCols returns the number of columns as integer.
func (m *Matrix) GetCols() int {
	return int(m.Cols)
}

// Show shows the Matrix.
func (m *Matrix) Show() {
	for y := 0; y < int(m.Rows); y++ {
		for x := 0; x < int(m.Cols); x++ {
			fmt.Printf("%f ", m.M[y][x])
		}
		fmt.Println()
	}
}

// Reshape changes the size of the matrix.
func (m *Matrix) Reshape(r, c uint) (*Matrix, error) {
	if r*c != m.Rows*m.Cols {
		return nil, fmt.Errorf("Size is not same")
	}
	row := 0
	col := 0
	newMat := make([][]float32, r)
	for y := 0; y < int(r); y++ {
		newMat[y] = make([]float32, c)
		for x := 0; x < int(c); x++ {
			if col > int(m.Cols)-1 {
				row++
				col = 0
			}
			newMat[y][x] = m.M[row][col]
			col++
		}
	}
	return NewMatrix(newMat), nil
}

func cmp(a, b []float32, resCh chan bool) {
	for i := range a {
		if a[i] != b[i] {
			resCh <- false
		}
	}
	resCh <- true
}

// Equals returns wheter the matrixes is same.
func (m *Matrix) Equals(t *Matrix) bool {
	if m.Rows != t.Rows || m.Cols != t.Cols {
		return false
	}
	ch := make(chan bool, m.Cols)
	for y := 0; y < int(m.Rows); y++ {
		go cmp(m.M[y], t.M[y], ch)
	}
	for y := 0; y < int(m.Rows); y++ {
		if res := <-ch; !res {
			return false
		}
	}
	return true
}

func execCalc(newMat [][]float32, y int, a, b *Matrix, mode CalcMode, wg *sync.WaitGroup) {
	newMat[y] = make([]float32, a.Cols)
	for x := 0; x < int(a.Cols); x++ {
		switch mode {
		case AddOP:
			newMat[y][x] = a.M[y][x] + b.M[y][x]
		case SubOP:
			newMat[y][x] = a.M[y][x] - b.M[y][x]
		}
	}
	wg.Done()
}

// Add adds Matrixces.
func Add(a, b *Matrix) (*Matrix, error) {
	if a.Cols != b.Cols || a.Rows != b.Rows {
		return nil, errors.New("Size was wrong")
	}
	res := make([][]float32, a.Rows)
	var wg sync.WaitGroup
	for y := 0; y < int(a.Rows); y++ {
		wg.Add(1)
		go execCalc(res, y, a, b, AddOP, &wg)
	}
	wg.Wait()
	return NewMatrix(res), nil
}

// Sub substracts Matrixces.
func Sub(a, b *Matrix) (*Matrix, error) {
	if a.Cols != b.Cols || a.Rows != b.Rows {
		return nil, errors.New("Size was wrong")
	}
	res := make([][]float32, a.Rows)
	var wg sync.WaitGroup
	for y := 0; y < int(a.Rows); y++ {
		wg.Add(1)
		go execCalc(res, y, a, b, SubOP, &wg)
	}
	wg.Wait()
	return NewMatrix(res), nil
}

func execMul(newMat [][]float32, i int, a, b *Matrix, wg *sync.WaitGroup) {
	newMat[i] = make([]float32, b.Cols)
	for j := 0; j < int(b.Cols); j++ {
		partial := float32(0.0)
		for k := 0; k < int(b.Rows); k++ {
			partial += a.M[i][k] * b.M[k][j]
		}
		newMat[i][j] = partial
	}
	wg.Done()
}

// Mul multiples Matrixces.
func Mul(a, b *Matrix) (*Matrix, error) {
	if a.Cols != b.Rows {
		return nil, errors.New("Size was wrong")
	}
	res := make([][]float32, a.Rows)
	var wg sync.WaitGroup
	for i := 0; i < int(a.Rows); i++ {
		wg.Add(1)
		go execMul(res, i, a, b, &wg)
	}
	wg.Wait()
	return NewMatrix(res), nil
}

func execElem(newMat [][]float32, y int, a, b *Matrix, wg *sync.WaitGroup, f func(a, b float32) float32) {
	newMat[y] = make([]float32, a.Cols)
	for x := 0; x < int(a.Cols); x++ {
		newMat[y][x] = f(a.M[y][x], b.M[y][x])
	}
	wg.Done()
}

// ElemAdd adds matrix elementwise.
func ElemAdd(a, b *Matrix) (*Matrix, error) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return nil, errors.New("Size was wrong")
	}
	res := make([][]float32, a.Rows)
	var wg sync.WaitGroup
	for y := 0; y < int(a.Rows); y++ {
		wg.Add(1)
		go execElem(res, y, a, b, &wg, func(a, b float32) float32 {
			return a + b
		})
	}
	wg.Wait()
	return NewMatrix(res), nil
}

// ElemSub substitutes matrix elementwise.
func ElemSub(a, b *Matrix) (*Matrix, error) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return nil, errors.New("Size was wrong")
	}
	res := make([][]float32, a.Rows)
	var wg sync.WaitGroup
	for y := 0; y < int(a.Rows); y++ {
		wg.Add(1)
		go execElem(res, y, a, b, &wg, func(a, b float32) float32 {
			return a - b
		})
	}
	wg.Wait()
	return NewMatrix(res), nil
}

// ElemMul multiples matrix elementwise.
func ElemMul(a, b *Matrix) (*Matrix, error) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return nil, errors.New("Size was wrong")
	}
	res := make([][]float32, a.Rows)
	var wg sync.WaitGroup
	for y := 0; y < int(a.Rows); y++ {
		wg.Add(1)
		go execElem(res, y, a, b, &wg, func(a, b float32) float32 {
			return a * b
		})
	}
	wg.Wait()
	return NewMatrix(res), nil
}

// ElemDiv divides matrix elementwise.
func ElemDiv(a, b *Matrix) (*Matrix, error) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return nil, errors.New("Size was wrong")
	}
	res := make([][]float32, a.Rows)
	var wg sync.WaitGroup
	for y := 0; y < int(a.Rows); y++ {
		wg.Add(1)
		go execElem(res, y, a, b, &wg, func(a, b float32) float32 {
			return a / b
		})
	}
	wg.Wait()
	return NewMatrix(res), nil
}

func (m *Matrix) flatten(res []float32, y int, wg *sync.WaitGroup) {
	for x := 0; x < int(m.Cols); x++ {
		res[y*int(m.Cols)+x] = m.M[y][x]
	}
	wg.Done()
}

// Flatten make Matrix flat.
func (m *Matrix) Flatten() []float32 {
	res := make([]float32, m.Rows*m.Cols)
	var wg sync.WaitGroup
	for y := 0; y < int(m.Rows); y++ {
		wg.Add(1)
		go m.flatten(res, y, &wg)
	}
	wg.Wait()
	return res
}

func (m *Matrix) execFunc(mat [][]float32, y int, wg *sync.WaitGroup, f func(float32, ...interface{}) float32, args ...interface{}) {
	mat[y] = make([]float32, m.Cols)
	for x := 0; x < int(m.Cols); x++ {
		mat[y][x] = f(m.M[y][x], args...)
	}
	wg.Done()
}

// BroadcastFunc executes function for bitwise.
func (m *Matrix) BroadcastFunc(f func(float32, ...interface{}) float32, args ...interface{}) *Matrix {
	newMatrix := make([][]float32, m.Rows)
	var wg sync.WaitGroup
	for y := 0; y < int(m.Rows); y++ {
		wg.Add(1)
		go m.execFunc(newMatrix, y, &wg, f, args...)
	}
	wg.Wait()
	return NewMatrix(newMatrix)
}

// BroadcastAdd adds matrix as bitwise.
func (m *Matrix) BroadcastAdd(v float32) *Matrix {
	return m.BroadcastFunc(func(e float32, a ...interface{}) float32 {
		return e + a[0].(float32)
	}, v)
}

// BroadcastSub substitute matrix as bitwise.
func (m *Matrix) BroadcastSub(v float32) *Matrix {
	return m.BroadcastFunc(func(e float32, a ...interface{}) float32 {
		return e - a[0].(float32)
	}, v)
}

// BroadcastMul multiples matrix as bitwise.
func (m *Matrix) BroadcastMul(v float32) *Matrix {
	return m.BroadcastFunc(func(e float32, a ...interface{}) float32 {
		return e * a[0].(float32)
	}, v)
}

// BroadcastDiv divide matrix as bitwise.
func (m *Matrix) BroadcastDiv(v float32) *Matrix {
	return m.BroadcastFunc(func(e float32, a ...interface{}) float32 {
		return e / a[0].(float32)
	}, v)
}

func (m *Matrix) zeroPad(rows, cols uint, newMatrix [][]float32, w uint, y int, wg *sync.WaitGroup) {
	newCols := m.Cols + w*2
	newMatrix[y] = make([]float32, newCols)
	for x := 0; x < int(newCols); x++ {
		if y > int(w)-1 && y < int(rows+w) && x > int(w)-1 && x < int(cols+w) {
			newMatrix[y][x] = m.M[y-int(w)][x-int(w)]
		} else {
			newMatrix[y][x] = 0.0
		}
	}
	wg.Done()
}

func (m *Matrix) edgePad(rows, cols uint, newMatrix [][]float32, w uint, y int, wg *sync.WaitGroup) {
	newCols := m.Cols + w*2
	newMatrix[y] = make([]float32, newCols)
	for x := 0; x < int(newCols); x++ {
		if y < int(w) && x < int(w) {
			newMatrix[y][x] = m.M[0][0]
		} else if y < int(w) && x > int(w)-1 && x < int(cols+w) {
			newMatrix[y][x] = m.M[0][x-int(w)]
		} else if y < int(w) && x > int(cols+w)-1 {
			newMatrix[y][x] = m.M[0][cols-1]
		} else if y > int(w)-1 && y < int(rows+w) && x < int(w) {
			newMatrix[y][x] = m.M[y-int(w)][0]
		} else if y > int(w)-1 && y < int(rows+w) && x > int(cols+w)-1 {
			newMatrix[y][x] = m.M[y-int(w)][cols-1]
		} else if y > int(rows+w)-1 && x < int(w) {
			newMatrix[y][x] = m.M[rows-1][0]
		} else if y > int(rows+w)-1 && x > int(w)-1 && x < int(cols+w) {
			newMatrix[y][x] = m.M[rows-1][x-int(w)]
		} else if y > int(rows+w)-1 && x > int(cols+w)-1 {
			newMatrix[y][x] = m.M[rows-1][cols-1]
		} else {
			newMatrix[y][x] = m.M[y-int(w)][x-int(w)]
		}
	}
	wg.Done()
}

// Pad pads the Matrix.
func (m *Matrix) Pad(w uint, mode PadMode) *Matrix {
	newRows := m.Rows + w*2
	newMatrix := make([][]float32, newRows)
	rows, cols := m.GetSize()
	var wg sync.WaitGroup
	for y := 0; y < int(newRows); y++ {
		wg.Add(1)
		switch mode {
		case Zero:
			go m.zeroPad(rows, cols, newMatrix, w, y, &wg)
		case Edge:
			go m.edgePad(rows, cols, newMatrix, w, y, &wg)
		}
	}
	wg.Wait()
	return NewMatrix(newMatrix)
}

// Dot calculate dot of two vector.
func Dot(v1, v2 []float32) (float32, error) {
	if len(v1) != len(v2) {
		return 0.0, fmt.Errorf("Length mismatched %d, %d\n", len(v1), len(v2))
	}
	sum := float32(0.0)
	for i := 0; i < len(v1); i++ {
		sum += v1[i] * v2[i]
	}
	return sum, nil
}

// Dot2d calculate dot of a matrix.
func Dot2d(m1, m2 [][]float32) (float32, error) {
	sum := float32(0.0)
	for i := 0; i < len(m1); i++ {
		partial, err := Dot(m1[i], m2[i])
		if err != nil {
			return 0.0, err
		}
		sum += partial
	}
	return sum, nil
}

// Slice2d slices a matrix.
func Slice2d(s [][]float32, rs, re, cs, ce uint) [][]float32 {
	sr := make([][]float32, re-rs)
	copy(sr, s[rs:re])
	for y := 0; y < len(sr); y++ {
		sr[y] = sr[y][cs:ce]
	}
	return sr
}

func (m *Matrix) execConv(newMat [][]float32, f *Matrix, y int, cols, rows, stride uint, errCh chan error) {
	newMat[y] = make([]float32, cols)
	var err error
	for x := 0; x < int(cols); x++ {
		newMat[y][x], err = Dot2d(Slice2d(m.M, uint(y)*stride, uint(y)*stride+f.Rows, uint(x)*stride, uint(x)*stride+f.Cols), f.M)
		if err != nil {
			errCh <- err
		}
	}
	errCh <- nil
}

// Convolve2d convolve a 2d matrix.
func (m *Matrix) Convolve2d(f *Matrix, stride, pad uint, mode PadMode) (*Matrix, error) {
	rows := (m.Rows-f.Rows+2*pad)/stride + 1
	cols := (m.Cols-f.Rows+2*pad)/stride + 1
	if pad > 0 {
		m = m.Pad(pad, mode)
	}
	newMat := make([][]float32, rows)
	errCh := make(chan error, cols)
	for y := 0; y < int(rows); y++ {
		go m.execConv(newMat, f, y, cols, rows, stride, errCh)
	}

	for i := 0; i < int(rows); i++ {
		err := <-errCh
		if err != nil {
			return nil, err
		}
	}
	close(errCh)
	return NewMatrix(newMat), nil
}

func maxPool(m [][]float32) float32 {
	max := m[0][0]
	for y := 0; y < len(m); y++ {
		for x := 0; x < len(m[0]); x++ {
			if max < m[y][x] {
				max = m[y][x]
			}
		}
	}
	return max
}

func avgPool(m [][]float32) float32 {
	sum := float32(0.0)
	for y := 0; y < len(m); y++ {
		for x := 0; x < len(m[0]); x++ {
			sum += m[y][x]
		}
	}
	return sum / float32(len(m)*len(m[0]))
}

func (m *Matrix) execPool(newMat [][]float32, y, rows, cols int, h, s uint, mode PoolingMode, wg *sync.WaitGroup) {
	newMat[y] = make([]float32, cols)
	for x := 0; x < cols; x++ {
		switch mode {
		case Max:
			newMat[y][x] = maxPool(Slice2d(m.M, uint(y)*s, uint(y)*s+h, uint(x)*s, uint(x)*s+h))
		case Avg:
			newMat[y][x] = avgPool(Slice2d(m.M, uint(y)*s, uint(y)*s+h, uint(x)*s, uint(x)*s+h))
		}
	}
	wg.Done()
}

// Pooling calculate pooling.
func (m *Matrix) Pooling(h, s uint, mode PoolingMode) *Matrix {
	rows := int(math.Ceil(float64(m.Rows-h)/float64(s))) + 1
	cols := int(math.Ceil(float64(m.Cols-h)/float64(s))) + 1
	if (m.Rows-h)%s != 0 {
		padded := NewMatrix(m.M).Pad(1, Zero)
		m.M = padded.M
		m.Rows = uint(len(m.M))
		m.Cols = uint(len(m.M[0]))
	}
	newMat := make([][]float32, rows)
	var wg sync.WaitGroup
	for y := 0; y < rows; y++ {
		wg.Add(1)
		go m.execPool(newMat, y, rows, cols, h, s, mode, &wg)
	}
	wg.Wait()
	return NewMatrix(newMat)
}

func (m *Matrix) t(newMat [][]float32, y int, wg *sync.WaitGroup) {
	col := make([]float32, m.Rows)
	for x := 0; x < int(m.Rows); x++ {
		col[x] = m.M[x][y]
	}
	newMat[y] = col
	wg.Done()
}

// T transports the Matrix.
func (m *Matrix) T() *Matrix {
	newMat := make([][]float32, m.Cols)
	var wg sync.WaitGroup
	for y := 0; y < int(m.Cols); y++ {
		wg.Add(1)
		go m.t(newMat, y, &wg)
	}
	wg.Wait()
	return NewMatrix(newMat)
}

func (m *Matrix) clip(newMat [][]float32, y int, start, end float32, wg *sync.WaitGroup) {
	newMat[y] = make([]float32, m.Cols)
	for x := 0; x < int(m.Cols); x++ {
		e := m.M[y][x]
		if e < start {
			newMat[y][x] = start
		} else if e > end {
			newMat[y][x] = end
		} else {
			newMat[y][x] = e
		}
	}
	wg.Done()
}

// Clip clips matrix.
func (m *Matrix) Clip(start, end float32) *Matrix {
	newMat := make([][]float32, m.Rows)
	var wg sync.WaitGroup
	for y := 0; y < int(m.Rows); y++ {
		wg.Add(1)
		go m.clip(newMat, y, start, end, &wg)
	}
	wg.Wait()
	return NewMatrix(newMat)
}

// SumVec calculates sum of the vector.
func SumVec(v []float32) float32 {
	sum := float32(0.0)
	for _, e := range v {
		sum += e
	}
	return sum
}

func makeCol(m [][]float32, colSize, rs, cs, kernelSize uint) []float32 {
	col := make([]float32, colSize)
	idx := 0
	for y := rs; y < rs+kernelSize; y++ {
		for x := cs; x < cs+kernelSize; x++ {
			col[idx] = m[y][x]
			idx++
		}
	}
	return col
}

// Im2Col make clumns matrix from matrix.
func (m *Matrix) Im2Col(kernelSize, stride uint) *Matrix {
	colSize := kernelSize * kernelSize
	var res [][]float32
	for y := 0; y < int(m.Rows-kernelSize+1); y += int(stride) {
		for x := 0; x < int(m.Cols-kernelSize+1); x += int(stride) {
			res = append(res, makeCol(m.M, colSize, uint(y), uint(x), kernelSize))
		}
	}
	return NewMatrix(res)
}
