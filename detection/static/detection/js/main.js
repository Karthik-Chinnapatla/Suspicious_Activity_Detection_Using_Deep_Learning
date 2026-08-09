document.addEventListener('DOMContentLoaded', () => {
    // 1. Drag and Drop Upload Dropzone Handler
    const dropzone = document.getElementById('upload_dropzone');
    const fileInput = document.getElementById('video_file_input');
    const fileNameDisplay = document.getElementById('selected_file_name');
    const fileSizeDisplay = document.getElementById('selected_file_size');
    const fileDetailsBox = document.getElementById('file_details_box');
    const uploadForm = document.getElementById('video_upload_form');
    const processingOverlay = document.getElementById('processing-overlay');

    if (dropzone && fileInput) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, () => dropzone.classList.add('dragover'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, () => dropzone.classList.remove('dragover'), false);
        });

        dropzone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files && files.length > 0) {
                fileInput.files = files;
                updateFileDetails(files[0]);
            }
        });

        dropzone.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                updateFileDetails(fileInput.files[0]);
            }
        });

        function updateFileDetails(file) {
            if (!file) return;
            if (fileNameDisplay) fileNameDisplay.textContent = file.name;
            if (fileSizeDisplay) {
                const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
                fileSizeDisplay.textContent = `${sizeMB} MB`;
            }
            if (fileDetailsBox) fileDetailsBox.style.display = 'block';
        }
    }

    // 2. Form submission spinner overlay
    if (uploadForm && processingOverlay) {
        uploadForm.addEventListener('submit', (e) => {
            if (fileInput && fileInput.files.length === 0) {
                return;
            }
            processingOverlay.style.display = 'flex';
        });
    }

    // 3. Auto dismiss alerts after 6 seconds
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            if (bsAlert) bsAlert.close();
        }, 6000);
    });

    // 4. Light / Dark Theme Toggle Handler
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    const themeToggleIcon = document.getElementById('theme-toggle-icon');

    function updateThemeUI(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        if (themeToggleIcon) {
            if (theme === 'light') {
                themeToggleIcon.className = 'fa-solid fa-sun text-warning';
            } else {
                themeToggleIcon.className = 'fa-solid fa-moon text-info';
            }
        }
    }

    const currentTheme = localStorage.getItem('cctv_theme') || 'dark';
    updateThemeUI(currentTheme);

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const activeTheme = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
            localStorage.setItem('cctv_theme', activeTheme);
            updateThemeUI(activeTheme);
        });
    }
});

