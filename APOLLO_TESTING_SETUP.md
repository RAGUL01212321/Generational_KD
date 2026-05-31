# Apollo Model Testing - Setup Complete ✓

Local testing environment for Apollo 0.5B and 2B models is ready!

## 🎯 Quick Start

### Option 1: Using the Quick Script
```bash
cd /home/ror-technologies/Hikigai/Gen_KD

# List available models
./run_apollo.sh list

# Test Apollo 0.5B
./run_apollo.sh test-0.5B

# Test Apollo 2B
./run_apollo.sh test-2B

# Interactive inference with Apollo 0.5B
./run_apollo.sh infer-0.5B

# Interactive inference with Apollo 2B
./run_apollo.sh infer-2B    
```

### Option 2: Direct Commands
```bash
cd /home/ror-technologies/Hikigai/Gen_KD

# Test model loading
python3 apollo_testing/test_models.py --model apollo_0.5B
python3 apollo_testing/test_models.py --model apollo_2B

# Interactive inference
python3 apollo_testing/inference.py --model apollo_0.5B
python3 apollo_testing/inference.py --model apollo_2B
```

## 📊 Available Models

### Apollo 0.5B
- **Base Model**: Qwen/Qwen2-0.5B
- **Parameters**: 494 million
- **Architecture**: Qwen2ForCausalLM
- **Hidden Size**: 1024
- **Layers**: 24
- **Attention Heads**: 16
- **Max Positions**: 32,768
- **Memory**: ~1-2 GB
- **Status**: ✓ Ready

### Apollo 2B
- **Base Model**: microsoft/phi-2
- **Parameters**: 2.78 billion
- **Architecture**: PhiForCausalLM
- **Hidden Size**: 2560
- **Layers**: 32
- **Attention Heads**: 32
- **Max Positions**: 2048
- **Memory**: ~4-6 GB
- **Status**: ✓ Ready

## 📁 Project Structure

```
/home/ror-technologies/Hikigai/Gen_KD/
├── apollo_testing/                    [NEW]
│   ├── __init__.py                   # Package init
│   ├── config.py                     # Model configurations
│   ├── loader.py                     # Model loading utilities
│   ├── test_models.py                # Test suite
│   ├── inference.py                  # Interactive inference
│   └── README.md                     # Detailed documentation
├── run_apollo.sh                     [NEW] # Quick start script
└── ...
```

## 🚀 Common Commands

### List Available Models
```bash
./run_apollo.sh list
python3 apollo_testing/test_models.py --list
```

### Test Model Loading
```bash
# Test specific model
./run_apollo.sh test-0.5B
./run_apollo.sh test-2B

# Test all models
./run_apollo.sh test-all

# Direct command
python3 apollo_testing/test_models.py --model apollo_0.5B --device cuda
```

### Interactive Inference
```bash
# Start interactive session
./run_apollo.sh infer-0.5B

# Then type your prompts:
# You: What is artificial intelligence?
# Apollo_0.5B: [generates response]

# Commands in interactive mode:
# - Any text: Generate response
# - quit/exit: Exit program
# - clear: Clear history
# - history: Show all prompts
```

### Programmatic Usage
```python
from apollo_testing import ApolloModelLoader

# Load and test model
with ApolloModelLoader("apollo_0.5B") as loader:
    loader.load_model()
    loader.print_model_info()
    
    # Generate text
    response = loader.generate(
        "Explain machine learning",
        max_length=100,
        temperature=0.7
    )
    print(response[0])
```

## 📈 Performance Tips

### For Faster Inference
```bash
# Use GPU (requires CUDA)
python3 apollo_testing/inference.py --model apollo_0.5B --device cuda

# Use smaller model
./run_apollo.sh infer-0.5B
```

### For Lower Memory Usage
```bash
# Use CPU
python3 apollo_testing/inference.py --model apollo_0.5B --device cpu

# Use smaller model
./run_apollo.sh infer-0.5B
```

### For Batch Processing
```python
from apollo_testing.loader import ApolloModelLoader

loader = ApolloModelLoader("apollo_0.5B")
loader.load_model()

prompts = [
    "What is AI?",
    "How does ML work?",
    "Explain deep learning"
]

for prompt in prompts:
    response = loader.generate(prompt, max_length=50)
    print(f"Q: {prompt}")
    print(f"A: {response[0]}\n")
```

## 🔧 Configuration

### Available Models
Models are configured in `apollo_testing/config.py`. Current models:

```python
APOLLO_MODELS = {
    "apollo_0.5B": ApolloModelConfig(
        model_id="Qwen/Qwen2-0.5B",
        architecture="Qwen2ForCausalLM",
        hidden_size=1024,
        num_hidden_layers=24,
        # ... more config
    ),
    "apollo_2B": ApolloModelConfig(
        model_id="microsoft/phi-2",
        architecture="PhiForCausalLM",
        hidden_size=2560,
        num_hidden_layers=32,
        # ... more config
    ),
}
```

### Add Custom Models
Edit `apollo_testing/config.py`:

```python
APOLLO_MODELS["custom_model"] = ApolloModelConfig(
    name="custom_model",
    model_id="huggingface/model-id",
    architecture="ModelArchitecture",
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    vocab_size=50257,
    max_position_embeddings=2048,
)
```

Then use:
```bash
python3 apollo_testing/test_models.py --model custom_model
python3 apollo_testing/inference.py --model custom_model
```

## 📚 Available Scripts

### `run_apollo.sh` - Quick Start Script
Convenient wrapper for common commands.

```bash
./run_apollo.sh test-0.5B    # Test loading
./run_apollo.sh infer-0.5B   # Interactive
./run_apollo.sh list         # List models
```

### `apollo_testing/test_models.py` - Test Suite
Comprehensive testing of models.

```bash
# Test specific model
python3 apollo_testing/test_models.py --model apollo_0.5B

# Test with inference
python3 apollo_testing/test_models.py --model apollo_2B --inference

# Test all models
python3 apollo_testing/test_models.py

# List models
python3 apollo_testing/test_models.py --list
```

### `apollo_testing/inference.py` - Interactive Inference
Real-time model testing interface.

```bash
# Start session
python3 apollo_testing/inference.py --model apollo_0.5B

# Specify device
python3 apollo_testing/inference.py --model apollo_2B --device cuda
```

### `apollo_testing/loader.py` - Core Library
Model loading and inference utilities for programmatic use.

```python
from apollo_testing.loader import ApolloModelLoader

loader = ApolloModelLoader("apollo_0.5B")
loader.load_model()
response = loader.generate("Hello")
```

### `apollo_testing/config.py` - Configuration
Model definitions and utilities.

```python
from apollo_testing.config import get_apollo_model_config, get_available_models

config = get_apollo_model_config("apollo_0.5B")
all_models = get_available_models()
```

## 🐛 Troubleshooting

### Issue: "Out of Memory"
**Solution:**
```bash
# Use smaller model
./run_apollo.sh infer-0.5B

# Use CPU instead of GPU
python3 apollo_testing/inference.py --model apollo_0.5B --device cpu

# Reduce generation length in interactive mode
```

### Issue: "Model not found"
**Solution:**
```bash
# Ensure internet connection
# Models are downloaded on first use from HuggingFace

# Or pre-download:
python3 apollo_testing/test_models.py --model apollo_0.5B
```

### Issue: CUDA not available
**Solution:**
```bash
# Check CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# Use CPU
python3 apollo_testing/inference.py --model apollo_0.5B --device cpu
```

### Issue: Slow inference
**Solution:**
```bash
# Use GPU if available (faster)
python3 apollo_testing/inference.py --model apollo_0.5B --device cuda

# Use smaller model (apollo_0.5B is faster than apollo_2B)
./run_apollo.sh infer-0.5B

# Reduce max_length for generation
```

## 📊 System Requirements

- **Python**: 3.12
- **PyTorch**: 2.12.0+cu130
- **Transformers**: 5.9.0
- **CUDA**: Available (but CPU also works)
- **GPU VRAM**: 
  - Apollo 0.5B: ~1-2 GB
  - Apollo 2B: ~4-6 GB
- **RAM**: ~8 GB minimum

## ✨ Features

✓ Easy model loading and testing
✓ Interactive inference interface
✓ Model information display
✓ Text generation
✓ Batch processing support
✓ GPU and CPU support
✓ Programmatic API
✓ Quick start scripts

## 🎯 Next Steps

1. **Test basic loading:**
   ```bash
   ./run_apollo.sh test-0.5B
   ./run_apollo.sh test-2B
   ```

2. **Try interactive inference:**
   ```bash
   ./run_apollo.sh infer-0.5B
   ```

3. **Explore programmatic usage:**
   ```python
   from apollo_testing import ApolloModelLoader
   # See README.md for examples
   ```

4. **Integrate with your pipeline:**
   - Use `ApolloModelLoader` in your training code
   - Reference `apollo_testing/config.py` for model configs
   - See examples in `apollo_testing/test_models.py`

## 📖 Documentation

- **Detailed Guide**: [apollo_testing/README.md](apollo_testing/README.md)
- **API Reference**: See docstrings in `apollo_testing/loader.py`
- **Configuration**: See `apollo_testing/config.py`

## 🔗 Resources

- [HuggingFace Models](https://huggingface.co/models)
- [Qwen2 Documentation](https://huggingface.co/Qwen)
- [Microsoft Phi Documentation](https://huggingface.co/microsoft)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

---

**Status**: ✅ Ready for local testing

**Created**: May 26, 2026
**Environment**: Linux / CUDA 130 / PyTorch 2.12.0
**GPU**: NVIDIA RTX 5090 (33.7GB)

**Happy testing!** 🚀
