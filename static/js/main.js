document.getElementById('microphone').addEventListener('click', function(){
    var msg = 'clicked!'
    console.log(msg)

    var url = "/process/"
        fetch(url, {
            method: 'POST',
            headers:{
                'Content-Type':'application/json',
                'X-CSRFToken' : csrftoken,
            },
            body:JSON.stringify(msg)
        })

        // .then((response) =>{
        // return response.json()
        // })

        .then((data) =>{
            console.log('data:', data)
        })
    })