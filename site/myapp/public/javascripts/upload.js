

let upload_uuid;
let dropzone_handler;
let num_sent = 0;
let queued_filenames;

let upload_input_format = /[\s `!@#$%^&*()+\=\[\]{};':"\\|,<>\/?~]/;


const FILE_FORMAT = /[\s `!@#$%^&*()+\=\[\]{};':"\\|,<>\/?~]/;
const IMAGE_SET_NAME_FORMAT = /[\s `!@#$%^&*()+\=\[\]{}.;':"\\|,<>\/?~]/;




function clear_form() {
    $("#image_set_name_input").val("");
    dropzone_handler.removeAllFiles();
}

function disable_input() {

    global_disabled = true;

    disable_blue_buttons(["upload_button"]);


    let inputs = ["image_set_name_input"];

    for (let input of inputs) {
        $("#" + input).prop("disabled", true);
        $("#" + input).css("opacity", 0.5);
    }

    disable_red_buttons(["remove_image_set_files"]);

    $("#image_set_dropzone").addClass("disabled_dropzone");
    $("#image_set_dropzone").css("opacity", 0.7);

    $("#modal_close").hide();

}



function enable_input() {

    enable_blue_buttons(["upload_button"]);

    let inputs = ["image_set_name_input"];

    for (let input of inputs) {
        $("#" + input).prop("disabled", false);
        $("#" + input).css("opacity", 1.0);
    }
    
    enable_red_buttons(["remove_image_set_files"]);

    $("#image_set_dropzone").removeClass("disabled_dropzone");
    $("#image_set_dropzone").css("opacity", 1.0);

    $("#modal_close").show();

}


function test_image_set_name() {
    let input_val = $("#image_set_name_input").val();
    let input_length = input_val.length;
    if (input_length == 0) {
        return [false, "An image set name must be provided."];
    }
    if (input_length < 3) {
        return [false, "The provided image set name is too short. At least 3 characters are required."];
    }
    if (input_length > 20) {
        return [false, "The provided image set name is too long. 20 characters is the maximum allowed length."];
    }
    if (IMAGE_SET_NAME_FORMAT.test(input_val)) {
        return [false, "The provided image set name contains invalid characters. White space and most special characters are not allowed."];
    }
    return [true, ""];
}



function remove_all_files() {
    if (!(global_disabled)) {
        dropzone_handler.removeAllFiles(true);
    }
}

function create_image_set_dropzone() {

    if (dropzone_handler) {
        dropzone_handler.destroy();
    }

    $("#dropzone_container").empty();

    $("#dropzone_container").append(
        `<table>` +
            `<tr>` +
                `<td style="width: 100%"></td>` +
                `<td>` +
                    `<div id="remove_image_set_files" class="button-red button-red-hover" style="width: 140px; font-size: 14px; padding: 2px; margin: 2px" onclick="remove_all_files()">` +
                        `<i class="fa-solid fa-circle-minus" style="padding-right: 5px"></i>` +
                            `Remove All Files` +
                    `</div>` +
                `</td>` +
            `</tr>` +
        `</table>` +
        `<div id="image_set_dropzone" class="dropzone" style="height: 180px">` +
            `<div class="dz-message data-dz-message">` +
                `<span>Drop Images Here</span>` +
            `</div>` +
            `<div id="image_set_upload_loader" class="loader" hidden></div>` +
        `</div>`
    );

    dropzone_handler = new Dropzone("#image_set_dropzone", { 
        url: ff_path + "image_set_upload",
        autoProcessQueue: false,
        paramName: function(n) { return 'source_file[]'; },
        uploadMultiple: true,
        farm_name: '',
        field_name: '',
        mission_date: '',
        parallelUploads: 10,
        maxUploads: 10000,
        maxFilesize: 450,
        addRemoveLinks: true,
        dictRemoveFile: "Remove File",
        dictCancelUpload: "",
    });

}

function add_dropzone_listeners() {

    dropzone_handler.on("success", function(file, response) {   

        dropzone_handler.removeFile(file);
        if (dropzone_handler.getAcceptedFiles().length == 0) {

            dropzone_handler.removeAllFiles(true);
            num_sent = 0;
            dropzone_handler.options.autoProcessQueue = false;

            let new_image_set_name = $("#image_set_name_input").val()
            image_set_names.push(new_image_set_name);
            // let selected_image_set_name = $("#image_set_combo").val();
            populate_image_set_combo();
            $("#image_set_combo").val(new_image_set_name).change(); //selected_image_set_name);
            global_disabled = false;

            // show_modal_message(`Success!`, `<div align="center">Your image set has been successfully uploaded.<br>The image set is now being processed.`);
            close_modal();

        }
    });

    dropzone_handler.on("error", function(file, response) {

        let upload_error;
        if (typeof(response) == "object" && "error" in response) {
            upload_error = response.error;
        }
        else {
            upload_error = response;
        }

    
        num_sent = 0;
        dropzone_handler.options.autoProcessQueue = false;
        dropzone_handler.removeAllFiles(true);
    
        show_modal_message(`Error`, upload_error);
        global_disabled = false;

    });

    dropzone_handler.on("addedfile", function() {

        if (dropzone_handler.options.autoProcessQueue) {
            let upload_error = "A file was added after the upload was initiated. Please ensure that all files have been added to the queue before pressing the 'Upload' button."
            dropzone_handler.removeAllFiles(true);
            num_sent = 0;
            dropzone_handler.options.autoProcessQueue = false;
        
            show_modal_message(`Error`, 
                upload_error
            );
            global_disabled = false;
        }
    });


    dropzone_handler.on('sending', function(file, xhr, formData) {

        formData.append('image_set_name', $("#image_set_name_input").val());
        formData.append("queued_filenames",  queued_filenames.join(","));
        if (num_sent == 0) {
            upload_uuid = uuidv4();
        }
        formData.append('upload_uuid', upload_uuid);
        num_sent++;
        formData.append("num_sent", num_sent.toString());

    });
}


function submit_upload() {

    disable_input();
    $("#upload_error_message").html("");
    $("#image_set_upload_loader").show();

    queued_filenames = [];

    let res;
    res = test_image_set_name();
    if (res[0]) {
        if (dropzone_handler.getQueuedFiles().length == 0) {
            res = [false, "At least one image must be provided."];
        }
    }
    if (res[0]) {
        for (let f of dropzone_handler.getQueuedFiles()) {
            if (FILE_FORMAT.test(f.name)) {
                res = [false, "One or more filenames contains illegal characters. White space and most special characters are not allowed."];
            }
        }
    }
    if (res[0]) {
        for (let f of dropzone_handler.getQueuedFiles()) {
            queued_filenames.push(f.name);
        }
        if (has_duplicates(queued_filenames)) {
            res = [false, "The image set contains duplicate filenames."];
        }
    }
    if (res[0]) {
        if (image_set_names.includes($("#image_set_name_input").val())) {
            res = [false, "Image set names must be unique! The provided name is already in use."];
        }
    }

    if (!(res[0])) {
        queued_filenames = [];
        $("#upload_error_message").html(res[1]);
        enable_input();
        global_disabled = false;
        $("#image_set_upload_loader").hide();
        return;
    }

    $("#image_set_dropzone").animate({ scrollTop: 0 }, "fast");
    dropzone_handler.options.autoProcessQueue = true;
    dropzone_handler.processQueue();

}



function show_upload_modal() {

    show_modal_message("Upload Image Set", 
    `<div>` +

        `<table>` +
            `<tr>` + 
                `<th>` + 
                    `<div class="table_head" style="width: 170px">Image Set Name</div>` + 
                `</th>` +
                `<th>` + 
                    `<div style="width: 250px">` +
                        `<input id="image_set_name_input" class="nonfixed_input">` +
                    `</div>` +
                `</th>` +
            `</tr>` +
        `</table>` + 
        `<div style="height: 10px"></div>` + 

        `<div id="dropzone_container" style="margin: 0px 20px; border: 1px solid white; border-radius: 5px"></div>` +

        `<div style="height: 10px"></div>` + 
        `<div style="text-align: center">` +
            `<button id="upload_button" class="button-blue button-blue-hover" style="width: 120px; height: 30px; font-size: 16px">` +
                `<i class="fa-solid fa-file-arrow-up" style="margin-right: 10px"></i>Upload` +
            `</button>` +
        `</div>` +
        `<div style="height: 10px"></div>` + 
        `<div style="text-align: center">` +
            `<div style="height: 10px; color: white" id="upload_error_message"></div>` +
        `</div>` +
    `</div>`, modal_width=730);


    create_image_set_dropzone();
    add_dropzone_listeners();


    global_disabled = false;

    $("#upload_button").click(function(e) {
        submit_upload();
    });
}