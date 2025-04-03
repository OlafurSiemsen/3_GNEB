package main

import (
	"log"
	"strconv"
	"strings"

	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func resize(f *data.Slice, arg string) {
	if f.N_images > 1 {
		log.Fatal("Resize has not been implemented for multiple images")
	}
	s := parseSize(arg)
	resized := data.Resample(f, s)
	*f = *resized
}

func parseSize(arg string) (size [3]int) {
	words := strings.Split(arg, "x")
	if len(words) != 3 {
		log.Fatal("resize: need N0xN1xN2 argument")
	}
	for i, w := range words {
		v, err := strconv.Atoi(w)
		util.FatalErr(err)
		size[i] = v
	}
	return
}
