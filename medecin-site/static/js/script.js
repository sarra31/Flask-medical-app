// script.js

// JavaScript for form validation and submission
document.getElementById('appointmentForm').addEventListener('submit', function(event) {
    // Check if all required fields are filled
    const inputs = this.querySelectorAll('input[required], select[required], textarea[required]');
    let allValid = true;

    inputs.forEach(function(input) {
        if (!input.value) {
            allValid = false;
            alert(`Veuillez remplir le champ : ${input.previousElementSibling.innerText}`);
        }
    });

    // If all fields are valid, submit the form
    if (allValid) {
        console.log("Formulaire soumis avec succès!");
        console.log("Nom :", document.getElementById('name').value);
        console.log("Date :", document.getElementById('date').value);
        console.log("Time :", document.getElementById('time').value);
        console.log("Email :", document.getElementById('email').value);
        console.log("Numéro de téléphone :", document.getElementById('phone').value);
        console.log("Genre :", document.getElementById('gender').value);
        console.log("Raison :", document.getElementById('reason').value);
        console.log("Photo :", document.getElementById('photo').value);

        // Optionally clear the form fields
        // this.reset();

        // Allow form to submit normally
    } else {
        event.preventDefault(); // Prevent form submission if not valid
    }
});
// Tuberculosis Form Validation
document.getElementById('tuberculosisForm').addEventListener('submit', function(event) {
    const inputs = this.querySelectorAll('input[required], select[required]');
    let allValid = true;

    inputs.forEach(function(input) {
        if (!input.value) {
            allValid = false;
            alert(`Veuillez remplir le champ : ${input.previousElementSibling.innerText}`);
        }
    });

    if (allValid) {
        console.log("Formulaire de tuberculose soumis avec succès!");
        console.log("Nom :", document.getElementById('name').value);
        console.log("Âge :", document.getElementById('age').value);
        console.log("Genre :", document.getElementById('gender').value);
        console.log("Photo :", document.getElementById('photo').value);
    } else {
        event.preventDefault(); // Prevent form submission if not valid
    }
});