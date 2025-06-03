package engine

import (
	"fmt"
	"path"
	"reflect"
	"slices"
	"strings"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/draw"
	"github.com/mumax/3/dump"
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/oommf"
	"github.com/mumax/3/util"
)

func init() {
	DeclFunc("Save", Save, "Save space-dependent quantity once, with auto filename")
	DeclFunc("SaveAs", SaveAs, "Save space-dependent quantity with custom filename")

	DeclLValue("FilenameFormat", &fformat{}, "printf formatting string for output filenames.")
	DeclLValue("OutputFormat", &oformat{}, "Format for data files: OVF1_TEXT, OVF1_BINARY, OVF2_TEXT or OVF2_BINARY")

	DeclROnly("OVF1_BINARY", OVF1_BINARY, "OutputFormat = OVF1_BINARY sets binary OVF1 output")
	DeclROnly("OVF2_BINARY", OVF2_BINARY, "OutputFormat = OVF2_BINARY sets binary OVF2 output")
	DeclROnly("OVF1_TEXT", OVF1_TEXT, "OutputFormat = OVF1_TEXT sets text OVF1 output")
	DeclROnly("OVF2_TEXT", OVF2_TEXT, "OutputFormat = OVF2_TEXT sets text OVF2 output")
	DeclROnly("DUMP", DUMP, "OutputFormat = DUMP sets text DUMP output")
	DeclFunc("Snapshot", Snapshot, "Save image of quantity")
	DeclFunc("SnapshotAs", SnapshotAs, "Save image of quantity with custom filename")
	DeclVar("SnapshotFormat", &SnapshotFormat, "Image format for snapshots: jpg, png or gif.")
}

var (
	FilenameFormat = "%s%06d"    // formatting string for auto filenames.
	SnapshotFormat = "jpg"       // user-settable snapshot format
	outputFormat   = OVF2_BINARY // user-settable output format
)

type fformat struct{}

func (*fformat) Eval() interface{}      { return FilenameFormat }
func (*fformat) SetValue(v interface{}) { DrainOutput(); FilenameFormat = v.(string) }
func (*fformat) Type() reflect.Type     { return reflect.TypeOf("") }

type oformat struct{}

func (*oformat) Eval() interface{}      { return outputFormat }
func (*oformat) SetValue(v interface{}) { DrainOutput(); outputFormat = v.(OutputFormat) }
func (*oformat) Type() reflect.Type     { return reflect.TypeOf(OutputFormat(OVF2_BINARY)) }

// Save once, with auto file name
func Save(q Quantity, ind_image_variadic ...int) {
	qname := NameOf(q)
	fname := autoFname(NameOf(q), outputFormat, autonum[qname])
	SaveAs(q, fname, ind_image_variadic...)
	autonum[qname]++
}

// Save under given file name (transparent async I/O).
func SaveAs(q Quantity, fname string, ind_image_variadic ...int) {

	if !strings.HasPrefix(fname, OD()) {
		fname = OD() + fname // don't clean, turns http:// in http:/
	}
	images_specified := len(ind_image_variadic) != 0 // Checks if the user specified images to save
	if M.N_images == 1 {                             // Just the normal MuMax way
		if path.Ext(fname) == "" {
			fname += ("." + SuffixFromOutputFormat[outputFormat])
		}
		buffer := ValueOf(q) // TODO: check and optimize for Buffer()
		defer cuda.Recycle(buffer)
		info := data.Meta{Time: Time, Name: NameOf(q), Unit: UnitOf(q), CellSize: MeshOf(q).CellSize()}
		data := buffer.HostCopy() // must be copy (async io)
		queOutput(func() { saveAs_sync(fname, data, info, outputFormat) })
	} else { // Save multiple images
		stored_magnetization_ptr := M.buffer_                  // We store a pointer to the original magnetization...
		for it_image := 0; it_image < M.N_images; it_image++ { // ...iterate over the images...
			if images_specified && !slices.Contains(ind_image_variadic, it_image) { // Skips images that weren't specified by user
				continue
			}
			M.buffer_ = stored_magnetization_ptr.SubSlice(it_image) // ...one at a time...
			t_fname := insertImageIndex(fname, it_image)
			t_buffer := ValueOf(q) // TODO: check and optimize for Buffer()
			defer cuda.Recycle(t_buffer)
			info := data.Meta{Time: Time, Name: NameOf(q), Unit: UnitOf(q), CellSize: MeshOf(q).CellSize()}
			t_data := t_buffer.HostCopy() // must be copy (async io)
			queOutput(func() { saveAs_sync(t_fname, t_data, info, outputFormat) })
		}
		M.buffer_ = stored_magnetization_ptr //...and then restore the original magnetization pointer
	}

}

// Save image once, with auto file name
func Snapshot(q Quantity, ind_image_variadic ...int) {
	qname := NameOf(q)
	fname := fmt.Sprintf(OD()+FilenameFormat+"."+SnapshotFormat, qname, autonum[qname])
	SnapshotAs(q, fname, ind_image_variadic...)
	autonum[qname]++
}

func SnapshotAs(q Quantity, fname string, ind_image_variadic ...int) {
	if !strings.HasPrefix(fname, OD()) {
		fname = OD() + fname // don't clean, turns http:// in http:/
	}
	if path.Ext(fname) == "" {
		fname += ("." + SuffixFromOutputFormat[outputFormat])
	}
	images_specified := len(ind_image_variadic) != 0 // Checks if the user specified images to save
	if M.N_images == 1 {
		s := ValueOf(q)
		defer cuda.Recycle(s)
		data := s.HostCopy() // must be copy (asyncio)
		queOutput(func() { snapshot_sync(fname, data) })
	} else {
		stored_magnetization_ptr := M.buffer_                  // We store a pointer to the original magnetization...
		for it_image := 0; it_image < M.N_images; it_image++ { // ...iterate over the images...
			if images_specified && !slices.Contains(ind_image_variadic, it_image) { // Skips images that weren't specified by user
				continue
			}
			M.buffer_ = stored_magnetization_ptr.SubSlice(it_image)
			t_fname := insertImageIndex(fname, it_image)
			s := ValueOf(q)
			defer cuda.Recycle(s)
			data := s.HostCopy() // must be copy (asyncio)
			queOutput(func() { snapshot_sync(t_fname, data) })
		}
		M.buffer_ = stored_magnetization_ptr //...and then restore the original magnetization pointer
	}
}

func insertImageIndex(i_filename string, ind_image int) string {
	var o_filename string
	t_ext := path.Ext(i_filename)
	if t_ext == "" {
		o_filename = (i_filename + imageIndexSuffix(ind_image) + "." + SuffixFromOutputFormat[outputFormat])
	} else {
		o_filename = (i_filename[:len(i_filename)-len(t_ext)-1] + imageIndexSuffix(ind_image) + t_ext)
	}
	return o_filename
}

func imageIndexSuffix(ind_image int) string {
	imageIndexFormat := "i%03d"
	return fmt.Sprintf(imageIndexFormat, ind_image)
}

// synchronous snapshot
func snapshot_sync(fname string, output *data.Slice) {
	f, err := httpfs.Create(fname)
	util.FatalErr(err)
	defer f.Close()
	draw.RenderFormat(f, output, "auto", "auto", arrowSize, path.Ext(fname))
}

// synchronous save
func saveAs_sync(fname string, s *data.Slice, info data.Meta, format OutputFormat) {
	f, err := httpfs.Create(fname)
	util.FatalErr(err)
	defer f.Close()

	switch format {
	case OVF1_TEXT:
		oommf.WriteOVF1(f, s, info, "text")
	case OVF1_BINARY:
		oommf.WriteOVF1(f, s, info, "binary 4")
	case OVF2_TEXT:
		oommf.WriteOVF2(f, s, info, "text")
	case OVF2_BINARY:
		oommf.WriteOVF2(f, s, info, "binary 4")
	case DUMP:
		dump.Write(f, s, info)
	default:
		panic("invalid output format")
	}

}

// Sets the outputformat, accessible in .go files
// Possible options: OVF1_TEXT, OVF1_BINARY, OVF2_TEXT, OVF2_BINARY, DUMP
// Note that the options are global constants, not strings
func SetOutputFormat(format OutputFormat) {
	outputFormat = format
}

type OutputFormat int

const (
	OVF1_TEXT OutputFormat = iota + 1
	OVF1_BINARY
	OVF2_TEXT
	OVF2_BINARY
	DUMP
)

var (
	SuffixFromOutputFormat = map[OutputFormat]string{
		OVF1_TEXT:   "ovf",
		OVF1_BINARY: "ovf",
		OVF2_TEXT:   "ovf",
		OVF2_BINARY: "ovf",
		DUMP:        "dump"}
)
