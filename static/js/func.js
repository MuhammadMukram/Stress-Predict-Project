const svm = document.getElementById('svm_form');
const decTree = document.getElementById('dectree_form');
function showDecTree() {
    svm.style.display = 'none';
    decTree.style.display = 'block';
}

function showSvm() {
    decTree.style.display = 'none';
    svm.style.display = 'block';
}

function showAlert() {
    const deleteAlert = document.getElementsByClassName('modaldel-container')[0];
    deleteAlert.style.display = 'block';
}

function hideAlert() {
    const deleteAlert = document.getElementsByClassName('modaldel-container')[0];
    deleteAlert.style.display = 'none';
}