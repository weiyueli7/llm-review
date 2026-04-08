# LLM Review: Enhancing Creative Writing via Blind Peer Review Feedback

This repository contains the code and dataset for our paper **"LLM Review: Enhancing Creative Writing via Blind Peer Review Feedback"**. 
We introduce a structured multi-agent framework inspired by the blind peer review process, demonstrating how iterative feedback and self-revision significantly elevate the creativity and quality of Large Language Model responses.

## Repository Structure

- **`datasets/`**: Contains our benchmark datasets, e.g., the fully curated SciFi-100 dataset.
- **`experiments/`**: Contains all experiment configurations and the main running scripts for the multi-agent peer review framework.

## Getting Started🚀

### Installation
To install the required dependencies, set up the conda environment:

```bash
conda create -n llm-review python=3.10
conda activate llm-review
pip install -r requirements.txt
```

## Run

To execute the multi-agent experiments, rely on the main evaluation script located in `experiments/multi_agent`:

```bash
cd experiments/multi_agent

# Run the LLM Review framework with the SciFi dataset:
python llm_creativity.py -c agent_roles_review_llama.json -d ../../datasets/SciFi/scientific_writing.json -r 5 -t SciFi-Review
```

For more comprehensive instructions and hyperparameter tuning, refer to the document [experiments/README.md](experiments/README.md).

## Citation
If you find our research or data helpful, please cite the paper:

```bibtex
@article{li2026llm,
 title={LLM Review: Enhancing Creative Writing via Blind Peer Review Feedback},
 author={Li, Weiyue and Song, Mingxiao and Shen, Zhenda and Zhao, Dachuan and Long, Yunfan and Li, Yi and Li, Yongce and Yang, Ruyi and Wang, Mengyu},
 journal={arXiv preprint arXiv:2601.08003},
 year={2026}
}
```
