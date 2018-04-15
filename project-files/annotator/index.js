jQuery(document).ready(function($) {

	var filenames = [] // the file names 
	var annotations = [] // 2d object that has the annotations for each part
	var currentIndex = 0


	// read All The Files and put them into filenames[]

	var dir = "assets/";
	var fileextension = ".png";
	var currentAnnotations = ""

	$.ajax({
    url: dir,
    success: function (data) {
        $(data).find("a:contains(" + fileextension + ")").each(function () {
        	var filename = this.href.replace(window.location.host, "").replace("http://", "");
        	filenames.push(filename)
        });
        
        // update the label
        readCheckboxes()
    	showAnnotationsLabel()
    }
    });

    // when checking a box

    $("#checkboxes input").change(function(event) {


    	// read checkboxes 
    	readCheckboxes()

		// update the label
    	showAnnotationsLabel() 

    });

    // When clicking on next

	$("#next_button").click(function(event) {

		if(currentIndex == filenames.length){
			return 
		}

		// update the index and go to next image
		
		currentIndex++

		// update ui
		updateUI()


	});

	// When clicking on prev

	$("#prev_button").click(function(event) {

		if(currentIndex == 0){
			return 
		}

		// upodate index

		currentIndex--

		// update ui
		updateUI()


    	
	});

	function oneIfTrue(value){
	if(value)
		return 1
	else
		return 0
	}

	function updateUI(){

		if(annotations[currentIndex] == null){
			readCheckboxes()
		}else{
			showCheckboxes()
		}
		showAnnotationsLabel()
		showImage()

	}

	function readCheckboxes(){

    	currentAnnotations = []

		$("#checkboxes input").each(function(index, el) {
			
			currentAnnotations.push(oneIfTrue($(this).is(":checked")))

		});

		annotations[currentIndex] = currentAnnotations
	}

	function showAnnotationsLabel(){
	   $("#current_annotation").text(filenames[currentIndex].substring(1)+" "+annotations[currentIndex].join()) 
	}

	function showImage(){
		$("#img").attr('src', "assets"+filenames[currentIndex]);
	}

	function showCheckboxes(){

		$("#checkboxes input").each(function(index, el) {
			if(annotations[currentIndex][index] == 1){
				$(this).attr('checked', 'true');
			}else{
				$(this).attr('checked', 'false');
				$(this).removeAttr('checked')
			}
		});
	}

});