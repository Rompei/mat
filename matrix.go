package matrix

import (
	"errors"
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// Matrix is object for matrix.
type Matrix struct {
	Rows uint
	Cols uint
	M    [][]float64
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
func NewMatrix(m [][]float64) *Matrix {
	return &Matrix{
		M:    m,
		Rows: uint(len(m)),
		Cols: uint(len(m[0])),
	}
}

// Zeros makes zero value matrix.
func Zeros(r, c uint) *Matrix {
	res := make([][]float64, r)
	for y := 0; y < int(r); y++ {
		res[y] = make([]float64, c)
	}
	return NewMatrix(res)
}

// Random returns random matrix.
func Random(r, c uint) *Matrix {
	rand.Seed(time.Now().UnixNano())
	res := make([][]float64, r)
	for y := 0; y < int(r); y++ {
		res[y] = make([]float64, c)
		for x := 0; x < int(c); x++ {
			res[y][x] = rand.Float64()
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
	newMat := make([][]float64, r)
	for y := 0; y < int(r); y++ {
		newMat[y] = make([]float64, c)
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

func cmp(a, b []float64, resCh chan bool) {
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

func execCalc(newMat [][]float64, y int, a, b *Matrix, mode CalcMode, wg *sync.WaitGroup) {
	newMat[y] = make([]float64, a.Cols)
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
	res := make([][]float64, a.Rows)
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
	res := make([][]float64, a.Rows)
	var wg sync.WaitGroup
	for y := 0; y < int(a.Rows); y++ {
		wg.Add(1)
		go execCalc(res, y, a, b, SubOP, &wg)
	}
	wg.Wait()
	return NewMatrix(res), nil
}

func execMul(newMat [][]float64, i int, a, b *Matrix, wg *sync.WaitGroup) {
	newMat[i] = make([]float64, a.Cols)
	for j := 0; j < int(b.Cols); j++ {
		partial := 0.0
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
	res := make([][]float64, a.Cols)
	var wg sync.WaitGroup
	for i := 0; i < int(a.Rows); i++ {
		wg.Add(1)
		go execMul(res, i, a, b, &wg)
	}
	wg.Wait()
	return NewMatrix(res), nil
}

func (m *Matrix) flatten(res []float64, y int, wg *sync.WaitGroup) {
	for x := 0; x < int(m.Cols); x++ {
		res[y*int(m.Cols)+x] = m.M[y][x]
	}
	wg.Done()
}

// Flatten make Matrix flat.
func (m *Matrix) Flatten() []float64 {
	res := make([]float64, m.Rows*m.Cols)
	var wg sync.WaitGroup
	for y := 0; y < int(m.Rows); y++ {
		wg.Add(1)
		go m.flatten(res, y, &wg)
	}
	wg.Wait()
	return res
}

func (m *Matrix) execFunc(mat [][]float64, y int, wg *sync.WaitGroup, f func(float64, ...interface{}) float64, args ...interface{}) {
	mat[y] = make([]float64, m.Cols)
	for x := 0; x < int(m.Cols); x++ {
		mat[y][x] = f(m.M[y][x], args...)
	}
	wg.Done()
}

// BroadcastFunc executes function for bitwise.
func (m *Matrix) BroadcastFunc(f func(float64, ...interface{}) float64, args ...interface{}) *Matrix {
	newMatrix := make([][]float64, m.Rows)
	var wg sync.WaitGroup
	for y := 0; y < int(m.Rows); y++ {
		wg.Add(1)
		go m.execFunc(newMatrix, y, &wg, f, args...)
	}
	wg.Wait()
	return NewMatrix(newMatrix)
}

// BroadcastAdd adds matrix as bitwise.
func (m *Matrix) BroadcastAdd(v float64) *Matrix {
	return m.BroadcastFunc(func(e float64, a ...interface{}) float64 {
		return e + a[0].(float64)
	}, v)
}

// BroadcastSub substitute matrix as bitwise.
func (m *Matrix) BroadcastSub(v float64) *Matrix {
	return m.BroadcastFunc(func(e float64, a ...interface{}) float64 {
		return e - a[0].(float64)
	}, v)
}

// BroadcastMul multiples matrix as bitwise.
func (m *Matrix) BroadcastMul(v float64) *Matrix {
	return m.BroadcastFunc(func(e float64, a ...interface{}) float64 {
		return e * a[0].(float64)
	}, v)
}

// BroadcastDiv divide matrix as bitwise.
func (m *Matrix) BroadcastDiv(v float64) *Matrix {
	return m.BroadcastFunc(func(e float64, a ...interface{}) float64 {
		return e / a[0].(float64)
	}, v)
}

func (m *Matrix) zeroPad(rows, cols uint, newMatrix [][]float64, w uint, y int, wg *sync.WaitGroup) {
	newCols := m.Cols + w*2
	newMatrix[y] = make([]float64, newCols)
	for x := 0; x < int(newCols); x++ {
		if y > int(w)-1 && y < int(rows+w) && x > int(w)-1 && x < int(cols+w) {
			newMatrix[y][x] = m.M[y-int(w)][x-int(w)]
		} else {
			newMatrix[y][x] = 0.0
		}
	}
	wg.Done()
}

func (m *Matrix) edgePad(rows, cols uint, newMatrix [][]float64, w uint, y int, wg *sync.WaitGroup) {
	newCols := m.Cols + w*2
	newMatrix[y] = make([]float64, newCols)
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
	newMatrix := make([][]float64, newRows)
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
func Dot(v1, v2 []float64) (float64, error) {
	if len(v1) != len(v2) {
		return 0.0, fmt.Errorf("Length mismatched %d, %d\n", len(v1), len(v2))
	}
	sum := 0.0
	for i := 0; i < len(v1); i++ {
		sum += v1[i] * v2[i]
	}
	return sum, nil
}

// Dot2d calculate dot of a matrix.
func Dot2d(m1, m2 [][]float64) (float64, error) {
	sum := 0.0
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
func Slice2d(s [][]float64, rs, re, cs, ce uint) [][]float64 {
	sr := make([][]float64, re-rs)
	copy(sr, s[rs:re])
	for y := 0; y < len(sr); y++ {
		sr[y] = sr[y][cs:ce]
	}
	return sr
}

func (m *Matrix) execConv(newMat [][]float64, f *Matrix, y int, cols, rows uint, errCh chan error) {
	newMat[y] = make([]float64, cols)
	var err error
	for x := 0; x < int(cols); x++ {
		newMat[y][x], err = Dot2d(Slice2d(m.M, uint(y), uint(y)+f.Rows, uint(x), uint(x)+f.Cols), f.M)
		if err != nil {
			errCh <- err
		}
	}
	errCh <- nil
}

// Convolve2d convolve a 2d matrix.
func (m *Matrix) Convolve2d(f *Matrix) (*Matrix, error) {
	rows := m.Rows - 2*(uint(f.Rows/2))
	cols := m.Cols - 2*(uint(f.Cols/2))
	newMat := make([][]float64, rows)
	errCh := make(chan error, cols)
	for y := 0; y < int(rows); y++ {
		go m.execConv(newMat, f, y, cols, rows, errCh)
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

func maxPool(m [][]float64) float64 {
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

func avgPool(m [][]float64) float64 {
	sum := 0.0
	for y := 0; y < len(m); y++ {
		for x := 0; x < len(m[0]); x++ {
			sum += m[y][x]
		}
	}
	return sum / float64(len(m)*len(m[0]))
}

func (m *Matrix) execPool(newMat [][]float64, y, rows, cols int, h, s uint, mode PoolingMode, wg *sync.WaitGroup) {
	newMat[y] = make([]float64, cols)
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
	rows := int((m.Rows-h)/s) + 1
	cols := int((m.Cols-h)/s) + 1
	newMat := make([][]float64, rows)
	var wg sync.WaitGroup
	for y := 0; y < rows; y++ {
		wg.Add(1)
		go m.execPool(newMat, y, rows, cols, h, s, mode, &wg)
	}
	wg.Wait()
	return NewMatrix(newMat)
}

func (m *Matrix) t(newMat [][]float64, y int, wg *sync.WaitGroup) {
	col := make([]float64, m.Rows)
	for x := 0; x < int(m.Rows); x++ {
		col[x] = m.M[x][y]
	}
	newMat[y] = col
	wg.Done()
}

// T transports the Matrix.
func (m *Matrix) T() *Matrix {
	newMat := make([][]float64, m.Cols)
	var wg sync.WaitGroup
	for y := 0; y < int(m.Cols); y++ {
		wg.Add(1)
		go m.t(newMat, y, &wg)
	}
	wg.Wait()
	return NewMatrix(newMat)
}

func (m *Matrix) clip(newMat [][]float64, y int, start, end float64, wg *sync.WaitGroup) {
	newMat[y] = make([]float64, m.Cols)
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
func (m *Matrix) Clip(start, end float64) *Matrix {
	newMat := make([][]float64, m.Rows)
	var wg sync.WaitGroup
	for y := 0; y < int(m.Rows); y++ {
		wg.Add(1)
		go m.clip(newMat, y, start, end, &wg)
	}
	wg.Wait()
	return NewMatrix(newMat)
}

// SumVec calculates sum of the vector.
func SumVec(v []float64) float64 {
	sum := 0.0
	for _, e := range v {
		sum += e
	}
	return sum
}
