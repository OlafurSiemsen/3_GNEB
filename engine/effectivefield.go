package engine

// Effective field

import "github.com/mumax/3/data"

var B_eff = NewVectorField("B_eff", "T", "Effective field", Curried_SetEffectiveField(&M))

// Nasty currying to avoid having to refactor function NewVectorField
func Curried_SetEffectiveField(mag *magnetization) func(dst *data.Slice) {
	f := func(dst *data.Slice) { SetEffectiveField(dst, mag) }
	return f
}

// Sets dst to the current effective field, in Tesla.
// This is the sum of all effective field terms,
// like demag, exchange, ...
func SetEffectiveField(dst *data.Slice, mag *magnetization) {
	if dst.N_images > 1 {
		n_images := dst.N_images
		stored_magnetization := M                         // We store a pointer to the original magnetization...
		for ind_img := 0; ind_img < n_images; ind_img++ { // ...iterate over the images...
			// Skips first and last images when FixEndImages is true
			if FixEndImages && (ind_img == 0 || ind_img == n_images-1) {
				continue
			}
			M = *stored_magnetization.SubMagnetization(ind_img) // TODO: Unnecessary?
			// M.buffer_ = stored_magnetization_ptr.SubSlice(ind_img) // ...one at a time...
			SetEffectiveField(dst.SubSlice(ind_img), mag.SubMagnetization(ind_img))
		}
		M = stored_magnetization //...and then restore the original magnetization pointer
		return
	}
	SetDemagField(dst, mag)     // set to B_demag...
	AddExchangeField(dst)       // ...then add other terms //TODO: Change signature, curry
	AddAnisotropyField(dst)     //TODO: Change signature, curry
	AddMagnetoelasticField(dst) //TODO: Change signature, curry
	B_ext.AddTo(dst)
	if !relaxing {
		B_therm.AddTo(dst)
	}
	AddCustomField(dst)
}
