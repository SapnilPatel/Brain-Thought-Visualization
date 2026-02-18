## Python scripts rewriting flow

### Phase-1
1. config.py
2. dataset.py
3. setup.py
4. stageA1_mbm_pretrain.py
5. sc_mbm/utils.py
6. sc_mbm/mae_for_fmri.py
7. sc_mbm/trainer.py
8. stageA2_mbm_finetune.py

### Phase-2
9. stageB_ldm_finetune.py
10. dc_ldm/ldm_for_fmri.py
11. dc_ldm/util.py
12. dc_ldm/models/autoencoder.py
13. * dc_ldm/models/diffusion/classifier.py 
14. dc_ldm/models/diffusion/ddim.py
15. * dc_ldm/models/diffusion/ddpm.py
16. dc_ldm/models/diffusion/plms.py
17. dc_ldm/modules/attention.py
18. dc_ldm/modules/ema.py
19. dc_ldm/modules/x_transformer.py
20. * dc_ldm/modules/diffusionmodules/model.py [line_number: 818]
21. dc_ldm/modules/diffusionmodules/openaimodel.py
22. dc_ldm/modules/diffusionmodules/util.py
23. dc_ldm/modules/distributions/distributions.py
24. - dc_ldm/modules/encoders/modules.py [Dependency: transformer]
25. - dc_ldm/modules/losses/contperceptual.py [Dependency: taming-transformers]
26. - dc_ldm/modules/losses/vqperceptual.py [Dependency: taming-transformers]

## Sequence to prepare working workstation
1. create main working directory: mkdir workstation
2. in the workstation clone github repo: git clone <repo_url>
3. create pretrains directory: mkdir workstation/pretrains
4. create assets directory: mkdir workstation/assets
5. create BOLD5000 directory in pretrains: mkdir pretrains/BOLD5000
6. create GOD directory in pretrains: mkdir pretrains/GOD
7. create ldm directory in pretrains: mkdir pretrains/ldm
8. create label2img directory in ldm: mkdir pretrains/ldm/label2img
