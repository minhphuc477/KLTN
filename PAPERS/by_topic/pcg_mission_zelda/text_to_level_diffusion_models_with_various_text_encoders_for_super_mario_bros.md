# Text-to-Level Diffusion Models With Various Text Encoders for Super Mario Bros

- PDF: [text_to_level_diffusion_models_with_various_text_encoders_for_super_mario_bros.pdf](../../text_to_level_diffusion_models_with_various_text_encoders_for_super_mario_bros.pdf)
- Topic: pcg_mission_zelda
- Reference IDs: arxiv:2507.00184

## Abstract / Core Idea
Recent research shows how diffusion models can unconditionally generate tile-based game levels, but use of diffusion models for text-to-level generation is underexplored. There are practical considerations for creating a usable model: caption/level pairs are needed, as is a text embedding model, and a way of generating entire playable levels, rather than individual scenes. We present strategies to automatically assign descriptive captions to an existing dataset, and train diffusion models using both pretrained text encoders and simple transformer models trained from scratch. Captions are automatically assigned to generated scenes so that the degree of overlap between input and output captions can be compared. We also assess the diversity and playability of the resulting level scenes. Results are compared with an unconditional diffusion model and a generative adversarial network, as well as the text-to-level approaches Five-Dollar Model and MarioGPT. Notably, the best diffusion model uses a simple transformer model for text embedding, and takes less time to train than diffusion models employing more complex text encoders, indicating that reliance on larger language models is not necessary. We also present a GUI allowing designers to construct long levels from model-generated scenes.

## Method Signals
- Keywords phat hien: diffusion

## Conclusion / Findings
with one gap. one descending staircase. one pipe. one irregular block cluster. full floor. full ceiling. one enemy. one coin. one irregular block cluster. a few towers. a few loose blocks. floor with one gap. a few enemies. one cannon. one tower. full floor. a few enemies. a few question blocks. one platform. one upside down pipe. two loose blocks. a few coin lines. one irregular block cluster. a few enemies. several coins. two ascending staircases. one question block. one rectangular block cluster. two cannons. floor with several gaps. two pipes. two enemies. one descending staircase. two towers. two upside down pipes. full floor. one descending staircase. one loose block. a few upside down pipes. full ceiling. two coins. one enemy. several platforms. two rectangular block clusters. one pipe. a few upside down pipes. floor with several gaps. one tower. MLM-negative0 0.86111111 0.63888889 0.68611111 1.0 0.75 −0.0138889 0.30555556 0.33888889 0.30555556 0.97222222 MiniLM-single-negative0 0.95833333 0.875 0.92222222 0.86111111 0.73611111 −0.0416667 0.56944444 0.31666667 0.47222222 0.63888889 MiniLM-multiple-negative0 0.97222222 0.63888889 0.93055556 0.83333333 0.76388889 −0.0416667 0.30555556 0.06944444 0.27777778 0.65277778 GTE-single-negative0 0.95833333 0.63888889 0.93611111 0.94444444 0.98611111 0.38888889 0.06944444 0.30555556 0.19444444 0.95833333 GTE-multiple-negative0 0.95833333 0.76388889 0.875 0.86111111 0.73611111 −0.2361111 0.40277778 0.30277778 0.38888889 0.65277778

## Relevance To KLTN
- Bai nay duoc xep vao nhom pcg_mission_zelda trong pipeline neural-symbolic topology-first.
- Dung de doi chieu voi khoi tuong ung trong docs va Chapter 3/4.
