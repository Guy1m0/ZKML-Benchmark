package common

import (
	"fmt"
	"testing"
	"unsafe"
)

func TestByteFloat(t *testing.T){
	a := 1.234
	ab := Float32ToBytes(float32(a), true)
	aa := BytesToFloat32(ab, true)
	fmt.Println(a, ab, aa)
}

func byteSliceToFloat32Slice(src []byte) []float32 {
	if len(src) == 0 {
		return nil
	}

	l := len(src) / 4
	ptr := unsafe.Pointer(&src[0])
	// It is important to keep in mind that the Go garbage collector
	// will not interact with this data, and that if src if freed,
	// the behavior of any Go code using the slice is nondeterministic.
	// Reference: https://github.com/golang/go/wiki/cgo#turning-c-arrays-into-go-slices
	return (*[1 << 26]float32)((*[1 << 26]float32)(ptr))[:l:l]
}

func encodeUnsafe(fs []float32) []byte {
    return unsafe.Slice((*byte)(unsafe.Pointer(&fs[0])), len(fs)*4)
}

func decodeUnsafe(bs []byte) []float32 {
    return unsafe.Slice((*float32)(unsafe.Pointer(&bs[0])), len(bs)/4)
}

func TestByteSliceToFloat32Slice(t *testing.T) {
	as := []float32{1.234, 2.345}
	asBytes := make([]byte, 0)
	for i := 0; i < len(as); i++ {
		asBytes = append(asBytes, Float32ToBytes(as[i], false)...)
	}
	fmt.Println(asBytes)
	fmt.Println(byteSliceToFloat32Slice(asBytes))
	fmt.Println(encodeUnsafe(as))
	fmt.Println(decodeUnsafe(encodeUnsafe(as)))
	fmt.Println(decodeUnsafe(asBytes))
}