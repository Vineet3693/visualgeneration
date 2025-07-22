
#!/bin/bash

# Deploy script for Free Visual AI Generator
# This script handles deployment to Streamlit Cloud and other platforms

set -e  # Exit on any error

echo "🚀 Starting deployment process for Free Visual AI Generator"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required files exist
check_requirements() {
    print_status "Checking deployment requirements..."
    
    required_files=(
        "requirements.txt"
        "src/streamlit_app/main.py"
        "config.yaml"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            print_error "Required file missing: $file"
            exit 1
        fi
    done
    
    print_success "All required files found"
}

# Optimize for deployment
optimize_for_deployment() {
    print_status "Optimizing for deployment..."
    
    # Run model optimization
    if [[ -f "scripts/optimize_models.py" ]]; then
        python scripts/optimize_models.py
        print_success "Models optimized"
    else
        print_warning "Model optimization script not found"
    fi
    
    # Check and optimize requirements.txt
    if [[ -f "requirements.txt" ]]; then
        print_status "Checking requirements.txt size..."
        
        # Count number of packages
        package_count=$(grep -c "==" requirements.txt || true)
        if [[ $package_count -gt 50 ]]; then
            print_warning "Large number of packages detected ($package_count). Consider optimizing."
        else
            print_success "Requirements.txt looks good ($package_count packages)"
        fi
    fi
}

# Create deployment-specific files
create_deployment_files() {
    print_status "Creating deployment-specific files..."
    
    # Create .streamlit/config.toml if it doesn't exist
    mkdir -p .streamlit
    
    if [[ ! -f ".streamlit/config.toml" ]]; then
        cat > .streamlit/config.toml << EOF
[global]
# Global configuration

[server]
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
EOF
        print_success "Created .streamlit/config.toml"
    fi
    
    # Create runtime.txt for Python version
    if [[ ! -f "runtime.txt" ]]; then
        python_version=$(python3 --version | cut -d' ' -f2)
        echo "python-$python_version" > runtime.txt
        print_success "Created runtime.txt with Python $python_version"
    fi
    
    # Create .gitignore if it doesn't exist
    if [[ ! -f ".gitignore" ]]; then
        cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env
.venv

# Models and cache
models/local_models/*/
*.pkl
*.pt
*.bin
*.safetensors

# Streamlit
.streamlit/secrets.toml

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Temporary files
tmp/
temp/
EOF
        print_success "Created .gitignore"
    fi
}

# Validate Streamlit app
validate_app() {
    print_status "Validating Streamlit app..."
    
    # Check if main.py can be imported
    cd src/streamlit_app
    if python -c "import main" 2>/dev/null; then
        print_success "Main app imports successfully"
    else
        print_error "Main app has import issues"
        cd ../..
        exit 1
    fi
    cd ../..
    
    # Check for common Streamlit issues
    if grep -r "st.experimental_rerun" src/ > /dev/null; then
        print_warning "Found deprecated st.experimental_rerun - consider updating to st.rerun"
    fi
    
    if grep -r "st.beta_" src/ > /dev/null; then
        print_warning "Found deprecated st.beta_ functions - consider updating"
    fi
}

# Deployment platform specific setup
setup_streamlit_cloud() {
    print_status "Setting up for Streamlit Cloud deployment..."
    
    # Check if GitHub repository is set up
    if [[ -d ".git" ]]; then
        print_success "Git repository detected"
        
        # Check if remote origin exists
        if git remote get-url origin > /dev/null 2>&1; then
            repo_url=$(git remote get-url origin)
            print_success "Remote repository: $repo_url"
        else
            print_warning "No remote repository configured"
            print_status "To deploy to Streamlit Cloud:"
            print_status "1. Push your code to GitHub"
            print_status "2. Visit https://share.streamlit.io/"
            print_status "3. Connect your GitHub repository"
            print_status "4. Set main file path to: src/streamlit_app/main.py"
        fi
    else
        print_warning "No Git repository found"
        print_status "Initialize Git repository for Streamlit Cloud deployment:"
        print_status "git init && git add . && git commit -m 'Initial commit'"
    fi
    
    # Create deployment info file
    cat > DEPLOYMENT.md << EOF
# Deployment Guide

## Streamlit Cloud Deployment

1. Push this repository to GitHub
2. Go to [https://share.streamlit.io/](https://share.streamlit.io/)
3. Click "New app"
4. Select your repository
5. Set the main file path to: \`src/streamlit_app/main.py\`
6. Click "Deploy!"

## Local Testing

\`\`\`bash
# Install requirements
pip install -r requirements.txt

# Download models (first time only)
python scripts/download_models.py

# Run the app
streamlit run src/streamlit_app/main.py
\`\`\`

## Configuration

- Main app: \`src/streamlit_app/main.py\`
- Configuration: \`config.yaml\`
- Requirements: \`requirements.txt\`
- Streamlit config: \`.streamlit/config.toml\`

## Features

- 🎨 Multiple visualization types
- 📝 Text-to-visual conversion
- 🖼️ Image analysis
- ⚡ Batch processing
- 📥 Multiple export formats
- 🆓 Completely free (no API keys required)
EOF
    
    print_success "Created DEPLOYMENT.md guide"
}

# Main deployment function
deploy() {
    echo "========================================"
    echo "🎨 Free Visual AI Generator Deployment"
    echo "========================================"
    
    check_requirements
    optimize_for_deployment
    create_deployment_files
    validate_app
    setup_streamlit_cloud
    
    print_success "Deployment preparation completed!"
    
    echo ""
    echo "📋 Next Steps:"
    echo "1. Commit and push your changes to GitHub"
    echo "2. Visit https://share.streamlit.io/ to deploy"
    echo "3. Set main file to: src/streamlit_app/main.py"
    echo "4. Your app will be live in a few minutes!"
    echo ""
    
    print_status "For local testing: streamlit run src/streamlit_app/main.py"
}

# Command line argument handling
case "${1:-deploy}" in
    "check")
        check_requirements
        ;;
    "optimize")
        optimize_for_deployment
        ;;
    "validate")
        validate_app
        ;;
    "deploy"|"")
        deploy
        ;;
    *)
        echo "Usage: $0 [check|optimize|validate|deploy]"
        echo "  check    - Check deployment requirements"
        echo "  optimize - Optimize for deployment"
        echo "  validate - Validate Streamlit app"
        echo "  deploy   - Full deployment preparation (default)"
        exit 1
        ;;
esac
