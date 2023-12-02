// async function queryprocessing() {
//     const query = document.getElementById("searchInput").value;
//     const results = await eel.search(query)();
//     document.getElementById("results").innerText = results;
// }

// document.getElementById("search").addEventListener("click",queryprocessing);



async function queryprocessing() {
    const searchInput = document.getElementById("searchInput");
    const query = searchInput.value.split(" ");
    var searchResult = {}
    if (query.length == 1) {
         searchResult = await eel.search(searchInput.value)();
    }
    else{
        searchResult = await eel.run_multiple_query(searchInput.value)();
    }
    
    console.log(searchResult);
    const searchResultDiv = document.getElementById("results")
    const tokenDiv = document.createElement("div");
    tokenDiv.innerHTML =  "Do you Mean: " + searchResult.token + "??<br><br>Documents!<br>";
    searchResultDiv.appendChild(tokenDiv);

    searchResult.ranked_documents.map((value, index) => {
        const resultDiv = document.createElement("div");
        resultDiv.innerHTML ="<br>" + value[1];
        // searchResultDiv.appendChild(resultDiv);
        
        let x = value[1];
        x = x.replace(".txt", "");
        console.log(x);

            // Create an anchor element
        let anchor = document.createElement("a");

        // Set the href attribute using the x value
        anchor.setAttribute("href","http://localhost:8000/content/" + x + ".html");

        // Set the target attribute (if needed)
        anchor.setAttribute("target", "_blank");
        anchor.textContent = x;

        // Append the anchor to the results div
        document.getElementById("results").appendChild(anchor);
        console.log(anchor)
        


    })
    const resultDiv = document.createElement("div")
    resultDiv.innerHTML =  "Precision: " + searchResult.Precision + "<br>" + "Recall: " + searchResult.Recall + "<br>" + "F Measure: " + searchResult.F_Measure
    searchResultDiv.appendChild(resultDiv)
    
}







document.getElementById("search").addEventListener("click",queryprocessing);
