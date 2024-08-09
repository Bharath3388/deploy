const selectImage = document.querySelector('.label');
const inputFile = document.querySelector('#img');
const imgArea = document.querySelector('.img-area');

selectImage.addEventListener('click', function (event) {
    event.preventDefault(); // Prevent default behavior of the label
    inputFile.click();
});

inputFile.addEventListener('change', function () {
    const image = this.files[0];
    if (image.size < 2000000) {
        const reader = new FileReader();
        reader.onload = () => {
            const allImg = imgArea.querySelectorAll('img');
            allImg.forEach(item => item.remove());
            const imgUrl = reader.result;
            const img = document.createElement('img');
            img.src = imgUrl;
            imgArea.appendChild(img);
            imgArea.classList.add('active');
            imgArea.dataset.img = image.name;
        };
        reader.readAsDataURL(image);
    } else {
        alert("Image size more than 2MB");
    }
});