package common

import (
	"bytes"
	"encoding/binary"
	"os"
	"reflect"
	"unsafe"
)

// vm only ===================================================================================

// memory layout in MIPS
const (
	INPUT_ADDR = 0x31000000
	OUTPUT_ADDR = 0x32000000
	MODEL_ADDR = 0x33000000
	MAGIC_ADDR = 0x30000800
)

func ByteAt(addr uint64, length int) []byte {
	var ret []byte
	bh := (*reflect.SliceHeader)(unsafe.Pointer(&ret))
	bh.Data = uintptr(addr)
	bh.Len = length
	bh.Cap = length
	return ret
}

// reading bytes from bigEndian or littleEndian
func ReadBytes(addr uint64, isBigEndian bool) []byte {
	rawSize := CopyBytes(ByteAt(addr, 4)) 
	size := BytesToInt32(rawSize, isBigEndian)
	ret := ByteAt(addr + 4, int(size)) 
	//shoud we copy here? may not for saving memory
	return ret
}

func Halt() {
	//os.Stderr.WriteString("THIS SHOULD BE PATCHED OUT\n")
	// the exit syscall is a jump to 0x5ead0000 now
	os.Exit(0)
}

func Output(output []byte, isBigEndian bool) {
	size := len(output)
	rawSize := IntToBytes(size,isBigEndian)
	mSize := ByteAt(OUTPUT_ADDR, 4)
	copy(mSize, rawSize)
	mData := ByteAt(OUTPUT_ADDR + 4, size)
	copy(mData, output)
	// magic code => have written the result
	magic := ByteAt(MAGIC_ADDR, 4)
	copy(magic, []byte{0x12, 0x34, 0x56, 0x78})
	// stop everything
	Halt()
}


func IntToBytes(n int, isBigEndian bool) []byte {
    x := int32(n)

    bytesBuffer := bytes.NewBuffer([]byte{})
	if isBigEndian {
		binary.Write(bytesBuffer, binary.BigEndian, x)
	} else {
		binary.Write(bytesBuffer, binary.LittleEndian, x)
	}
    return bytesBuffer.Bytes()
}

func BytesToInt32(b []byte, isBigEndian bool) int32 {
    bytesBuffer := bytes.NewBuffer(b)

    var x int32
	if isBigEndian {
		binary.Read(bytesBuffer, binary.BigEndian, &x)
	} else {
		binary.Read(bytesBuffer, binary.LittleEndian, &x)
	}
    

    return x
}

func Float32ToBytes(x float32, isBigEndian bool) []byte {
	bytesBuffer := bytes.NewBuffer([]byte{})
	if isBigEndian {
		binary.Write(bytesBuffer, binary.BigEndian, x)
	} else {
		binary.Write(bytesBuffer, binary.LittleEndian, x)
	}
	return bytesBuffer.Bytes()
}

func BytesToFloat32(b []byte, isBigEndian bool) float32 {
	byteBuffer := bytes.NewBuffer(b)
	var x float32 
	if isBigEndian {
		binary.Read(byteBuffer, binary.BigEndian, &x)
	} else {
		binary.Read(byteBuffer, binary.LittleEndian, &x)
	}
	
	return x
}

// CopyBytes returns an exact copy of the provided bytes.
func CopyBytes(b []byte) (copiedBytes []byte) {
	if b == nil {
		return nil
	}
	copiedBytes = make([]byte, len(b))
	copy(copiedBytes, b)

	return
}

// read from index then return the result and the next index
func ReadInt32FromBytes(data []byte, index *int, isBigEndian bool) (uint32) {
	if (*index + 4 > len(data)) {
		*index = len(data)
		return 0
	}
	buf := CopyBytes(data[*index:*index+4])
	ret := BytesToInt32(buf, isBigEndian)
	*index = *index + 4
	return uint32(ret)
}

func ReadFP32FromBytes(data []byte, index *int, isBigEndian bool) (float32) {
	if (*index + 4 > len(data)) {
		*index = len(data)
		return 0
	}
	buf := CopyBytes(data[*index:*index+4])
	ret := BytesToFloat32(buf, isBigEndian)
	*index = *index + 4
	return ret
}