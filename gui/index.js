// Image ID list (based on files in results folder)
const imageIds = [
    '20042', '20084', '20095', '20117', '20118', '20121',
    '20125', '20130', '20142', '20158', '20190', '20209',
    '20217', '20231', '20259', '20264', '20334', '20392',
    '20403', '20416', '20439', '20457', '20486', '20517',
    '20523', '20546', '20564', '20599', '20600', '20632',
    '20665', '20669', '20672', '20682', '20685', '20747',
    '20765', '20772', '20785', '20804', '20821', '20829',
    '20848', '20857', '20870', '20881', '20882', '20892',
    '20902', '20912', '20922', '20952'
];

// Current image index
let currentIndex = 0;

// DOM elements
const currentIdElement = document.getElementById('current-id');
const originalImgElement = document.getElementById('original-img');
const thresh01ImgElement = document.getElementById('thresh01-img');
const thresh03ImgElement = document.getElementById('thresh03-img');
const thresh05ImgElement = document.getElementById('thresh05-img');
const originalFilenameElement = document.getElementById('original-filename');
const thresh01FilenameElement = document.getElementById('thresh01-filename');
const thresh03FilenameElement = document.getElementById('thresh03-filename');
const thresh05FilenameElement = document.getElementById('thresh05-filename');
const prevBtn = document.getElementById('prev-btn');
const nextBtn = document.getElementById('next-btn');
const resetBtn = document.getElementById('reset-btn');
const idListElement = document.getElementById('id-list');
const modal = document.getElementById('image-modal');
const modalImg = document.getElementById('modal-img');
const modalCaption = document.getElementById('modal-caption');
const closeModal = document.querySelector('.close-modal');

// Initialize the application
function init() {
    // Load the first image
    loadCurrentImage();
    
    // Generate ID buttons
    generateIdButtons();
    
    // Set up event listeners
    setupEventListeners();
}

// Load current image based on currentIndex
function loadCurrentImage() {
    const currentId = imageIds[currentIndex];
    
    // Update current ID display
    currentIdElement.textContent = currentId;
    
    // Set image paths
    originalImgElement.src = `original/${currentId}.png`;
    thresh01ImgElement.src = `results/${currentId}_thresh0.1.png`;
    thresh03ImgElement.src = `results/${currentId}_thresh0.3.png`;
    thresh05ImgElement.src = `results/${currentId}_thresh0.5.png`;
    
    // Set filenames
    originalFilenameElement.textContent = `${currentId}.png`;
    thresh01FilenameElement.textContent = `${currentId}_thresh0.1.png`;
    thresh03FilenameElement.textContent = `${currentId}_thresh0.3.png`;
    thresh05FilenameElement.textContent = `${currentId}_thresh0.5.png`;
    
    // Update ID button states
    updateIdButtons();
}

// Generate ID navigation buttons
function generateIdButtons() {
    idListElement.innerHTML = '';
    
    imageIds.forEach((id, index) => {
        const button = document.createElement('button');
        button.className = 'id-btn';
        if (index === currentIndex) {
            button.classList.add('active');
        }
        button.textContent = id;
        button.dataset.index = index;
        
        button.addEventListener('click', () => {
            currentIndex = parseInt(button.dataset.index);
            loadCurrentImage();
        });
        
        idListElement.appendChild(button);
    });
}

// Update ID button active state
function updateIdButtons() {
    const buttons = document.querySelectorAll('.id-btn');
    buttons.forEach((button, index) => {
        if (index === currentIndex) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });
}

// Set up event listeners
function setupEventListeners() {
    // Previous button
    prevBtn.addEventListener('click', () => {
        currentIndex = (currentIndex - 1 + imageIds.length) % imageIds.length;
        loadCurrentImage();
    });
    
    // Next button
    nextBtn.addEventListener('click', () => {
        currentIndex = (currentIndex + 1) % imageIds.length;
        loadCurrentImage();
    });
    
    // Reset to first button
    resetBtn.addEventListener('click', () => {
        currentIndex = 0;
        loadCurrentImage();
    });
    
    // Image click to enlarge
    [originalImgElement, thresh01ImgElement, thresh03ImgElement, thresh05ImgElement].forEach(img => {
        img.addEventListener('click', function() {
            modal.style.display = 'block';
            modalImg.src = this.src;
            modalCaption.textContent = this.alt;
        });
    });
    
    // Close modal
    closeModal.addEventListener('click', () => {
        modal.style.display = 'none';
    });
    
    // Close modal when clicking outside
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });
    
    // Keyboard controls
    document.addEventListener('keydown', (e) => {
        switch(e.key) {
            case 'ArrowLeft':
                currentIndex = (currentIndex - 1 + imageIds.length) % imageIds.length;
                loadCurrentImage();
                break;
            case 'ArrowRight':
            case ' ':
                currentIndex = (currentIndex + 1) % imageIds.length;
                loadCurrentImage();
                break;
            case 'Home':
                currentIndex = 0;
                loadCurrentImage();
                break;
            case 'Escape':
                modal.style.display = 'none';
                break;
        }
    });
}

// Image error handling
[originalImgElement, thresh01ImgElement, thresh03ImgElement, thresh05ImgElement].forEach(img => {
    img.addEventListener('error', function() {
        this.src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300" viewBox="0 0 300 300"><rect width="100%" height="100%" fill="%23f0f0f0"/><text x="50%" y="50%" font-family="Arial" font-size="16" text-anchor="middle" fill="%23999" dy=".3em">Image Load Error</text></svg>';
        this.alt = 'Failed to load image';
    });
});

// Initialize when page loads
document.addEventListener('DOMContentLoaded', init);