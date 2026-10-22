CUDA_VISIBLE_DEVICES=0 python run_FLIP.py \
        --data_dir EPSERI/model_setup  \
        --measurement_file EPSERI/demo_data/-eps-Eri/full_im/target_roll001_coadded.npy  \
        --reference_file EPSERI/demo_data/-del-Eri/full_im/reference_dither1_coadded.npy \
        --opd_filename observation_opd.npy \
        --scene_name EPSERI \
        --stage_fit_flat_fields_start_iter 3500 \
        --stage_fit_flat_fields_end_iter 5500 \
        --px_mask_file EPSERI/model_setup/pixel_masks/pixel_mask_empty.npy \
        --ref_which_int 0 --sci_which_int 0 \
        --exp_name "EPSERI_test" \
        --iters 25000 --sci_targ_name epsEri \
        --ref_cutoff_iter 5500 --sci_lr_weight 0.005  \
        --inj_xpos 17 --inj_ypos 5 --inject_how_many_sigma_flux 30 --track_injected_planet\
        --measurement_noisemap EPSERI/demo_data/-eps-Eri/noise_maps/noisemap_target_roll001_coadded.npy \
        --lr 0.000000004 --stage_fluxpos_cutoff_iter 600 


