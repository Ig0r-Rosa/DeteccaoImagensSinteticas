# Bases do projeto (v3 social — sem CIFAKE no treino)

| Pasta | Papel | Como obter |
|---|---|---|
| `instagram_ai/` | Imagens **FAKE** de redes sociais | [Harvard Dataverse MGJPBL](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MGJPBL) |
| `celeba_real/` | Retratos **REAL** | [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) → `img_align_celeba.zip` |
| `landscape_real/` | Cenas **REAL** diversas (forte recomendado) | LHQ / COCO / Open Images / Unsplash |
| `archive/` | CIFAKE | **Não usada** no treino v3 atual |

O notebook `Cod/main_v3_instagram.ipynb` usa só as fontes sociais, treina na **imagem inteira** (letterbox 224×224) e salva `svm_v3_*.pkl`.
