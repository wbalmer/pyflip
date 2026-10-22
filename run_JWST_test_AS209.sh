CUDA_VISIBLE_DEVICES=0 python run_FLIP.py \
        --data_dir AS209_F335M/model_setup  \
        --measurement_file AS209_F335M/demo_data/EM-AS-209/full_im/target_roll002_coadded.npy  \
        --reference_file AS209_F335M/demo_data/HD-153135/full_im/reference_dither1_coadded.npy \
        --opd_filename observation_opd.npy \
        --scene_name AS209 \
        --stage_fit_flat_fields_start_iter 3500 \
        --stage_fit_flat_fields_end_iter 5500 \
        --px_mask_file AS209_F335M/model_setup/pixel_masks/pixel_mask_empty.npy \
        --ref_which_int 0 --sci_which_int 0 \
        --exp_name "AS209_F335M_test" \
        --iters 15000 --sci_targ_name EM-AS-209 \
        --ref_cutoff_iter 5500 --sci_lr_weight 0.005  \
        --forced_stop_iter 11000 \
        --inj_xpos 17 --inj_ypos 5 --inject_how_many_sigma_flux 30 --track_injected_planet\
        --measurement_noisemap AS209_F335M/demo_data/EM-AS-209/noise_maps/noisemap_target_roll002_coadded.npy \
        --lr 0.000000004 --stage_fluxpos_cutoff_iter 600 


