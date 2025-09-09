// DOM Elements
const fileInput = document.getElementById('fileInput');
const fileInputLabel = document.querySelector('.file-input-label');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const removePreview = document.getElementById('removePreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const uploadForm = document.getElementById('uploadForm');

// Section elements
const uploadSection = document.getElementById('uploadSection');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');

// Results elements
const medicineName = document.getElementById('medicineName');
const expiryDate = document.getElementById('expiryDate');
const aiAnalysisContent = document.getElementById('aiAnalysisContent');
const resultImage = document.getElementById('resultImage');

// Button elements
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const retryBtn = document.getElementById('retryBtn');

// File input change handler
fileInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        showPreview(file);
        analyzeBtn.disabled = false;
    }
});

// Show image preview
function showPreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        previewContainer.style.display = 'block';
        fileInputLabel.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Remove preview
removePreview.addEventListener('click', function() {
    previewContainer.style.display = 'none';
    fileInputLabel.style.display = 'inline-flex';
    fileInput.value = '';
    analyzeBtn.disabled = true;
});

// Form submission
uploadForm.addEventListener('submit', function(e) {
    e.preventDefault();
    
    const file = fileInput.files[0];
    if (!file) {
        showError('Please select an image file.');
        return;
    }
    
    showLoading();
    uploadFile(file);
});

// Show loading state
function showLoading() {
    hideAllSections();
    loadingSection.style.display = 'block';
    
    // Animate loading steps
    const steps = document.querySelectorAll('.step');
    let currentStep = 0;
    
    const stepInterval = setInterval(() => {
        if (currentStep < steps.length) {
            steps[currentStep].classList.add('active');
            currentStep++;
        } else {
            clearInterval(stepInterval);
        }
    }, 1000);
}

// Upload file to server
async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showResults(data);
        } else {
            showError(data.error || 'Analysis failed. Please try again.');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showError('Network error. Please check your connection and try again.');
    }
}

// Show results
function showResults(data) {
    hideAllSections();
    resultsSection.style.display = 'block';
    
    // Update medicine information
    medicineName.textContent = data.medicine_name;
    expiryDate.textContent = data.expiry_date;
    
    // Update AI analysis
    aiAnalysisContent.innerHTML = formatAIAnalysis(data.medicine_info);
    
    // Removed OCR results panel
    
    // Update result image
    resultImage.src = data.image;
}

// Format AI analysis content
function formatAIAnalysis(analysis) {
    if (!analysis) return '<p>No analysis available.</p>';
    
    // Split by ** markers and format
    const parts = analysis.split(/\*\*(.*?)\*\*/);
    let formatted = '';
    
    for (let i = 0; i < parts.length; i++) {
        if (i % 2 === 0) {
            // Regular text
            formatted += parts[i];
        } else {
            // Bold text (headers)
            formatted += `<strong>${parts[i]}</strong>`;
        }
    }
    
    // Replace line breaks with <br> tags
    formatted = formatted.replace(/\n/g, '<br>');
    
    return formatted;
}

// Format OCR results
// Removed OCR results formatter

// Show error
function showError(message) {
    hideAllSections();
    errorSection.style.display = 'block';
    document.getElementById('errorMessage').textContent = message;
}

// Hide all sections
function hideAllSections() {
    uploadSection.style.display = 'none';
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
}

// New analysis button
newAnalysisBtn.addEventListener('click', function() {
    resetForm();
    hideAllSections();
    uploadSection.style.display = 'block';
});

// Retry button
retryBtn.addEventListener('click', function() {
    resetForm();
    hideAllSections();
    uploadSection.style.display = 'block';
});

// Reset form
function resetForm() {
    fileInput.value = '';
    previewContainer.style.display = 'none';
    fileInputLabel.style.display = 'inline-flex';
    analyzeBtn.disabled = true;
    
    // Reset loading steps
    const steps = document.querySelectorAll('.step');
    steps.forEach(step => step.classList.remove('active'));
}

// Drag and drop functionality
const uploadCard = document.querySelector('.upload-card');

uploadCard.addEventListener('dragover', function(e) {
    e.preventDefault();
    uploadCard.classList.add('drag-over');
});

uploadCard.addEventListener('dragleave', function(e) {
    e.preventDefault();
    uploadCard.classList.remove('drag-over');
});

uploadCard.addEventListener('drop', function(e) {
    e.preventDefault();
    uploadCard.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            fileInput.files = files;
            showPreview(file);
            analyzeBtn.disabled = false;
        } else {
            showError('Please select a valid image file.');
        }
    }
});

// Add drag over styles
const style = document.createElement('style');
style.textContent = `
    .upload-card.drag-over {
        border: 2px dashed #667eea;
        background: rgba(102, 126, 234, 0.1);
    }
`;
document.head.appendChild(style);

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    // Set initial state
    analyzeBtn.disabled = true;
    
    // Add some interactive effects
    const cards = document.querySelectorAll('.info-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.transition = 'transform 0.3s ease';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
});

// Add smooth scrolling for better UX
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});
