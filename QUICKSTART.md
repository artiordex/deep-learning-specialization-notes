# Quick Start Guide

Welcome! This guide will help you get started quickly with the Deep Learning Specialization repository.

## 🎯 5-Minute Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the setup script:**
   ```bash
   python setup.py
   ```

3. **Try the PyTorch example:**
   ```bash
   python labs/pytorch_example.py
   ```

4. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

5. **Open the lab template:**
   - Navigate to `labs/lab_template.ipynb`
   - Start experimenting!

## 📖 Learning a New Concept

When learning a new concept from the course:

1. **Watch** the lecture video
2. **Take notes** using `notes/note_template.md`
3. **Implement** in a lab notebook (use `labs/lab_template.ipynb`)
4. **Experiment** with the concept using `experiments/experiment_template.md`

## 🔧 Common Tasks

### Create a new lab notebook

```bash
cp labs/lab_template.ipynb labs/course1-neural-networks/my_lab.ipynb
jupyter notebook labs/course1-neural-networks/my_lab.ipynb
```

### Run an experiment

```bash
# Create experiment directory
mkdir experiments/my_experiment
cp experiments/experiment_template.md experiments/my_experiment/experiment.md

# Edit and implement your experiment
```

### Use utilities in your code

```python
import sys
sys.path.append('..')
from utils import normalize_data, train_model, plot_training_curves

# Now use the utilities
data = normalize_data(raw_data)
```

## 💡 Tips

- **Use virtual environments** to keep dependencies isolated
- **Commit often** if you're tracking your own learning
- **Experiment freely** - the templates are just starting points
- **Document as you go** - future you will thank you!

## 🆘 Troubleshooting

### Import errors
```bash
# Make sure you're in the repository root
cd /path/to/deep-learning-specialization-notes

# Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### GPU not detected
```python
import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
print(torch.version.cuda)  # Check CUDA version
```

### Jupyter kernel not found
```bash
python -m ipykernel install --user --name dl-specialization
```

## 📚 Resources

- [Main README](README.md) - Comprehensive documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Course Website](https://www.deeplearning.ai/courses/deep-learning-specialization/)

## ✅ Checklist for First Session

- [ ] Install all dependencies
- [ ] Run setup.py successfully
- [ ] Run pytorch_example.py
- [ ] Open and run lab_template.ipynb
- [ ] Create your first note using note_template.md
- [ ] Bookmark this repository for easy access

---

Ready to dive in? Start with Course 1! 🚀
