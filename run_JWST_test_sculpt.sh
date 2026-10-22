CUDA_VISIBLE_DEVICES=0 python run_FLIP.py \
        --data_dir sculptor_test/model_setup  \
        --measurement_file sculptor_test/demo_data/HD-145964/full_im/target_roll134_coadded.npy  \
        --reference_file sculptor_test/demo_data/HD-145964-PSF/full_im/reference_dither1_coadded.npy \
        --opd_filename observation_opd.npy \
        --scene_name HD-145964 \
        --stage_fit_flat_fields_start_iter 3500 \
        --stage_fit_flat_fields_end_iter 5500 \
        --px_mask_file sculptor_test/model_setup/pixel_masks/pixel_mask_empty.npy \
        --ref_which_int 0 --sci_which_int 0 \
        --exp_name "HD-145964_test" \
        --iters 15000 --sci_targ_name HD-145964 \
        --ref_cutoff_iter 5500 --sci_lr_weight 0.005  \
        --forced_stop_iter 11000 \
        --inj_xpos 17 --inj_ypos 5 --inject_how_many_sigma_flux 0 --track_injected_planet\
        --measurement_noisemap sculptor_test/demo_data/HD-145964/noise_maps/noisemap_target_roll134_coadded.npy \
        --lr 0.000000004 --stage_fluxpos_cutoff_iter 600 


