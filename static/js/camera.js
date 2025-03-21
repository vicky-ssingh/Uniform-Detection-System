$(document).ready(function() {
    // Variables for video streaming and capture
    let video = document.getElementById('video');
    let canvas = document.getElementById('canvas');
    let context = canvas.getContext('2d');
    let streaming = false;

    // Handle file upload form submission
    $('#uploadForm').on('submit', function(e) {
        e.preventDefault();

        let fileInput = $('#imageUpload')[0];
        if (fileInput.files.length === 0) {
            alert('Please select an image to upload');
            return;
        }

        let formData = new FormData();
        formData.append('file', fileInput.files[0]);

        // Show loading state
        $('#uploadForm button').prop('disabled', true).html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...');

        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                if (response.status === 'success') {
                    displayResult(response);
                } else {
                    alert('Error: ' + response.message);
                }
                $('#uploadForm button').prop('disabled', false).text('Check Uniform');
            },
            error: function() {
                alert('An error occurred during the upload.');
                $('#uploadForm button').prop('disabled', false).text('Check Uniform');
            }
        });
    });

    // Start camera button
    $('#startCamera').on('click', function() {
        if (streaming) {
            stopCamera();
            $(this).text('Start Camera');
            $('#captureImage').hide();
            $('#video').hide();
        } else {
            startCamera();
            $(this).text('Stop Camera');
        }
    });

    // Capture image button
    $('#captureImage').on('click', function() {
        captureImage();
    });

    // Train model button
    $('#trainModel').on('click', function() {
        if (confirm('Are you sure you want to retrain the model? This may take several minutes.')) {
            $('#trainModel').prop('disabled', true).html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Training...');
            $('#trainingStatus').show().html('<div class="alert alert-info">Model training in progress. This may take several minutes. Please wait...</div>');

            $.ajax({
                url: '/train_model',
                type: 'GET',
                success: function(response) {
                    if (response.status === 'success') {
                        $('#trainingStatus').html('<div class="alert alert-success">Model trained successfully!</div>');
                    } else {
                        $('#trainingStatus').html('<div class="alert alert-danger">Error: ' + response.message + '</div>');
                    }
                    $('#trainModel').prop('disabled', false).text('Re-Train Model');
                },
                error: function() {
                    $('#trainingStatus').html('<div class="alert alert-danger">An error occurred during model training.</div>');
                    $('#trainModel').prop('disabled', false).text('Re-Train Model');
                }
            });
        }
    });

    // Cleanup images button
    $('#cleanupImages').on('click', function() {
        if (confirm('Are you sure you want to delete all uploaded and captured images?')) {
            $('#cleanupImages').prop('disabled', true).html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Deleting...');
            $('#cleanupStatus').show().html('<div class="alert alert-info">Deleting images. Please wait...</div>');

            $.ajax({
                url: '/cleanup',
                type: 'GET',
                success: function(response) {
                    if (response.status === 'success') {
                        $('#cleanupStatus').html('<div class="alert alert-success">' + response.message + '</div>');
                    } else {
                        $('#cleanupStatus').html('<div class="alert alert-danger">Error: ' + response.message + '</div>');
                    }
                    $('#cleanupImages').prop('disabled', false).text('Delete All Images');

                    // Hide the result card if it's visible
                    $('#resultCard').hide();
                },
                error: function() {
                    $('#cleanupStatus').html('<div class="alert alert-danger">An error occurred during cleanup.</div>');
                    $('#cleanupImages').prop('disabled', false).text('Delete All Images');
                }
            });
        }
    });

    // Function to start the camera
    function startCamera() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(function(stream) {
                    video.srcObject = stream;
                    video.play();
                    streaming = true;
                    $('#video').show();
                    $('#captureImage').show();
                })
                .catch(function(err) {
                    console.log("An error occurred: " + err);
                    alert("Cannot access camera. Please ensure camera permissions are granted.");
                });
        } else {
            alert("Your browser does not support camera access.");
        }
    }

    // Function to stop the camera
    function stopCamera() {
        if (streaming) {
            let stream = video.srcObject;
            let tracks = stream.getTracks();

            tracks.forEach(function(track) {
                track.stop();
            });

            video.srcObject = null;
            streaming = false;
        }
    }

    // Function to capture image from camera
    function captureImage() {
        if (streaming) {
            // Draw the video frame to the canvas
            context.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Convert canvas to blob
            canvas.toBlob(function(blob) {
                let formData = new FormData();
                formData.append('image', blob, 'capture.jpg');

                // Show loading state
                $('#captureImage').prop('disabled', true).html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...');

                $.ajax({
                    url: '/capture',
                    type: 'POST',
                    data: formData,
                    processData: false,
                    contentType: false,
                    success: function(response) {
                        if (response.status === 'success') {
                            $('#canvas').show();
                            displayResult(response);
                        } else {
                            alert('Error: ' + response.message);
                        }
                        $('#captureImage').prop('disabled', false).text('Capture Image');
                    },
                    error: function() {
                        alert('An error occurred during image processing.');
                        $('#captureImage').prop('disabled', false).text('Capture Image');
                    }
                });
            }, 'image/jpeg', 0.95);
        }
    }

    // Function to display the detection result
    function displayResult(response) {
        let result = response.result;
        let isUniform = result.is_uniform;
        let confidence = result.confidence.toFixed(2);

        // Update the result card
        $('#resultImage').attr('src', response.image_path);

        if (isUniform) {
            $('#resultStatus').text('Uniform Detected ✓').removeClass('uniform-danger').addClass('uniform-success');
            $('#actionMessage').removeClass('alert-danger').addClass('alert-success').text('Access Granted: Student is wearing proper uniform.');
        } else {
            $('#resultStatus').text('Non-Uniform Detected ✗').removeClass('uniform-success').addClass('uniform-danger');
            $('#actionMessage').removeClass('alert-success').addClass('alert-danger').text('Access Denied: Student is not wearing proper uniform.');
        }

        // Update confidence bar
        $('#confidenceBar').css('width', confidence + '%').attr('aria-valuenow', confidence).text(confidence + '%');
        $('#confidenceText').text('Confidence: ' + confidence + '%');

        if (confidence > 80) {
            $('#confidenceBar').removeClass('bg-warning bg-danger').addClass('bg-success');
        } else if (confidence > 50) {
            $('#confidenceBar').removeClass('bg-success bg-danger').addClass('bg-warning');
        } else {
            $('#confidenceBar').removeClass('bg-success bg-warning').addClass('bg-danger');
        }

        // Show the result card
        $('#resultCard').show();
    }
});
