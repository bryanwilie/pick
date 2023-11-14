# PICK: Polished & Informed Candidate Scoring for Knowledge-Grounded Dialogue Systems

This is the repo for the paper: [PICK: Polished & Informed Candidate Scoring for Knowledge-Grounded Dialogue Systems](https://arxiv.org/pdf/2309.10413.pdf). This framework addresses the key challenges in knowledge-grounded dialogue systems, such as hallucination and lack of coherence, through a generation re-scoring framework that empowers models to generate faithful and relevant responses without requiring additional labeled data or model tuning. Further details could be found [in the paper](https://arxiv.org/pdf/2309.10413.pdf).

## Steps:
1. Make sure all requirements are installed, or install it via: `pip install -r requirements.txt`
2. Prepare the dataset:
    - Download the wizard_of_wikipedia dataset:
        - `wget -P data_pool/wizard_of_wikipedia http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz`
        - `tar -xvzf data_pool/wizard_of_wikipedia/wizard_of_wikipedia.tgz -C data_pool/wizard_of_wikipedia/`
        - `rm -rf data_pool/wizard_of_wikipedia/wizard_of_wikipedia.tgz`
3. Prepare caffeinated_pandas to help in parallelization:
    - Download caffeinated-pandas repo to this repo in your local using:
        - `git clone https://github.com/scollay/caffeinated-pandas.git`
        - `mv caffeinated-pandas caffeinated_pandas`
3. Finetune your model using `run_ft_*.sh`
4. Do inference with your model using `run_eval_*.sh`
5. Score your generations further with other metrics, i.e. [FED](https://github.com/Shikib/fed.git), by cloning it to your local.
    
## Citation

This work is published at AACL-IJCNLP 2023 and you can find the details [in the paper](https://arxiv.org/pdf/2309.10413.pdf) (the link to AACL2023 paper is still currently not yet ready). Please cite our work if you find it useful.
```
@inproceedings{wilie2023pick,
  author    = {Wilie, Bryan  and  Xu, Yan  and  Chung, Willy  and  
              Cahyawijaya, Samuel  and  Lovenia, Holy  and  Fung, Pascale},
  title     = {PICK: Polished \& Informed Candidate Scoring for Knowledge-Grounded Dialogue Systems},
  booktitle = {Proceedings of the 13th International Joint Conference on Natural Language Processing 
                 and the 3rd Conference of the Asia-Pacific Chapter of 
                 the Association for Computational Linguistics},
  month     = {November},
  year      = {2023},
  address   = {Nusa Dua, Bali},
  publisher = {Association for Computational Linguistics},
  pages     = {980--995}
}
```
