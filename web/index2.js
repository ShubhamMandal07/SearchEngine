async function runinverted(){
    document.getElementById("results").style.display = "block";
    const  result = await eel.run_Inverted()();
    updateTable(result);
}

async function runpermuterm(){
    document.getElementById("results").style.display = "block";
    const  result = await eel.run_Permuterm()();
    updateTable(result);
}

async function runbiagram(){
    document.getElementById("results").style.display = "block";
    const  result = await eel.run_Biagram()();
    updateTable(result);
}
async function runsoundex(){
    document.getElementById("results").style.display = "block";
    const  result = await eel.run_Soundex()();
    updateTable(result);
}

function updateTable(output,button) {
    const outputTableBody = document.getElementById("outputTableBody");
    console.log(button)
    if (button == "inverted")
    {
        document.getElementById("col1").innerHTML = "Tokens";
        document.getElementById("col2").innerHTML = "Document Frequency ";
        document.getElementById("col3").innerHTML = "Posting List";

    }
    else if (button == "biagram")
    {
        document.getElementById("col1").innerHTML = "Bigram Index";
        document.getElementById("col2").innerHTML = "Tokens In Bigram";
        document.getElementById("col3").innerHTML = "";

    }
    else if (button == "permuterm")
    {
        document.getElementById("col1").innerHTML = "Permuterm Key";
        document.getElementById("col2").innerHTML = "Permuterm Value";
        document.getElementById("col3").innerHTML = "";

    }
    else if (button == "soundex")
    {
        document.getElementById("col1").innerHTML = "Soundex Code";
        document.getElementById("col2").innerHTML = "Tokens in Soundex";
        document.getElementById("col3").innerHTML = "";

    }


    // Clear existing rows
    outputTableBody.innerHTML = '';

    // Loop through the output data and add rows to the table
    output.forEach((item, index) => {
        const newRow = outputTableBody.insertRow();
        const serialCell = newRow.insertCell(0);
        const tokenCell = newRow.insertCell(1);
        const freqCell = newRow.insertCell(2);
        const postingCell = newRow.insertCell(3);

        serialCell.textContent = index + 1; // Serial No.
        tokenCell.textContent = item.token;
        freqCell.textContent = item.frequency;
        postingCell.textContent = item.posting_list;
    });
}

document.getElementById("inverted").addEventListener("click",runinverted);
document.getElementById("biagram").addEventListener("click",runbiagram);
document.getElementById("soundex").addEventListener("click",runsoundex);
document.getElementById("permuterm").addEventListener("click",runpermuterm);