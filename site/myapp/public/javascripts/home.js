
let image_set_names;
let overlay_appearance;

let viewer;
let predictions;
let cur_image_name;
let cur_image_names;

let img_nav_enabled = true;

let object_classes = ["Broken", "Not Fusarium", "Shriveled", "Tombstone"];



function viewer_onRedraw() {

    if (predictions == null || (!(cur_image_name in predictions))) {
        return;
    }

    let cur_pred_cls_idx = $("#pred_class_select").val();

    let viewer_bounds = viewer.viewport.getBounds();
    let container_size = viewer.viewport.getContainerSize();

    let hw_ratio = overlay.imgHeight / overlay.imgWidth;
    let min_x = Math.floor(viewer_bounds.x * overlay.imgWidth);
    let min_y = Math.floor((viewer_bounds.y / hw_ratio) * overlay.imgHeight);
    let viewport_w = Math.ceil(viewer_bounds.width * overlay.imgWidth);
    let viewport_h = Math.ceil((viewer_bounds.height / hw_ratio) * overlay.imgHeight);
    let max_x = min_x + viewport_w;
    let max_y = min_y + viewport_h;



    overlay.context2d().font = "14px arial";

    if (!($("#image_visible_switch").is(":checked"))) {
        let viewer_point_1 = viewer.viewport.imageToViewerElementCoordinates(
            new OpenSeadragon.Point(0, 0));
        let viewer_point_2 = viewer.viewport.imageToViewerElementCoordinates(
                new OpenSeadragon.Point(overlay.imgWidth, overlay.imgHeight));
                
        overlay.context2d().fillStyle = "#222621";         
        overlay.context2d().fillRect(
            viewer_point_1.x - 10,
            viewer_point_1.y - 10,
            (viewer_point_2.x - viewer_point_1.x) + 20,
            (viewer_point_2.y - viewer_point_1.y) + 20,
        );
    }


    

    overlay.context2d().lineWidth = 2;
    let added_inds = [];
    for (let i = 0; i < predictions[cur_image_name]["boxes"].length; i++) {

        let box = predictions[cur_image_name]["boxes"][i];
        let cls = predictions[cur_image_name]["classes"][i];

        if ((cur_pred_cls_idx != -1) && 
            (cls != cur_pred_cls_idx)) {
            continue;
        }

        let visible = (((box[1] < max_x) && (box[3] > min_x)) && ((box[0] < max_y) && (box[2] > min_y)));
        if (!visible) {
            continue;
        }

        overlay.context2d().strokeStyle = overlay_appearance["colors"][cls];
        overlay.context2d().fillStyle = overlay_appearance["colors"][cls] + "55";


        let viewer_point = viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(box[1], box[0]));
        let viewer_point_2 = viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(box[3], box[2]));

        overlay.context2d().strokeRect(
            viewer_point.x,
            viewer_point.y,
            (viewer_point_2.x - viewer_point.x),
            (viewer_point_2.y - viewer_point.y)
        );

        added_inds.push(i);
    }


    if ($("#scores_switch").is(":checked")) {
        for (let i = 0; i < added_inds.length; i++) {

            let ind = added_inds[i];
            let box = predictions[cur_image_name]["boxes"][ind];
            let score = predictions[cur_image_name]["classifier_scores"][ind];
        
            let box_width_pct_of_image = (box[3] - box[1]) / overlay.imgWidth;
            let disp_width = (box_width_pct_of_image / viewer_bounds.width) * container_size.x;
            let box_height_pct_of_image = (box[3] - box[1]) / overlay.imgHeight;
            let disp_height = (box_height_pct_of_image / viewer_bounds.height) * container_size.y;

            if ((disp_width * disp_height) < 10) {
                continue;
            }

            let viewer_point = viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(box[1], box[0]));
            let score_text = (Math.ceil(score * 100) / 100).toFixed(2);

            overlay.context2d().fillStyle = "#212326";
            overlay.context2d().fillRect(
                    viewer_point.x - 1,
                    viewer_point.y - 19,
                    36,
                    20
                );

            overlay.context2d().fillStyle = "white";
            overlay.context2d().fillText(score_text, 

                viewer_point.x + 3,
                viewer_point.y - 5
            );

        }
    }
}

function create_viewer(dzi_image_paths) {
    viewer = null;
    $("#seadragon_viewer").empty();
    viewer = OpenSeadragon({
        id: "seadragon_viewer",
        sequenceMode: true,
        prefixUrl: ff_path + "osd/images/",
        tileSources: dzi_image_paths,
        showNavigator: false,
        maxZoomLevel: 1000,
        zoomPerClick: 1,
        animationTime: 0.0,
        zoomPerScroll: 1.2,
        pixelsPerArrowPress: 0,
        nextButton: "next-btn",
        previousButton: "prev-btn",
        showNavigationControl: false,
        imageSmoothingEnabled: false,
    });

    viewer.innerTracker.keyDownHandler = null;
    viewer.innerTracker.keyPressHandler = null;
    viewer.innerTracker.keyHandler = null;

    overlay = viewer.canvasOverlay({
        clearBeforeRedraw: true
    });
    overlay.onRedraw = viewer_onRedraw;
    overlay.onOpen = function() {
        viewer.viewport.goHome();
    };
}


function apply_overlay_appearance_change() {

    let make_default = $("#make_colors_default").is(":checked");
    overlay_appearance["colors"] = [];
    for (let i = 0; i < object_classes.length; i++) {
        let overlay_color = $("#overlay_color_" + i).val();
        overlay_appearance["colors"].push(overlay_color);
    }
    if (make_default) {

        $.post($(location).attr('href'),
        {
            action: "save_overlay_appearance",
            overlay_appearance: JSON.stringify(overlay_appearance)
        },
        function(response, status) {
    
            if (response.error) {
                show_modal_message(`Error`, response.message);
            }
            else {
                apply_front_end_appearance_change();
            }
        });
    }
    else {
        apply_front_end_appearance_change();
    }
}

function apply_front_end_appearance_change() {

    if (viewer) {
        update_count_chart();
        viewer.raiseEvent('update-viewport');
    }
    close_modal();
}

function show_customize_overlays_modal() {

    show_modal_message(`Change Overlay Colors`,

        `<table id="color_table" style="border: 1px solid white; border-radius: 10px; padding: 20px 50px"></table>` +

        `<div style="height: 20px"></div>` +
        `<table>` +
            `<tr>` +
                `<td>` +
                    `<div class="em-text" style="width: 200px; text-align: right; font-size: 14px">Save Current Settings</div>` +
                `</td>` +
                `<td>` +
                    `<div style="width: 100px; text-align: left; margin-top: -5px">` +
                        `<label for="make_colors_default" class="container" style="display: inline; margin-left: 12px">` +
                            `<input type="checkbox" id="make_colors_default" name="make_colors_default">` +
                            `<span class="checkmark"></span>` +
                        `</label>` +
                    `</div>` +
                `</td>` +
            `</tr>` +
        `</table>` +

        `<table>` +
            `<tr>` +
                `<td>` +
                    `<button class="button-blue button-blue-hover" onclick="apply_overlay_appearance_change()" style="width: 120px; margin-top: 15px">Apply</button>` +
                `</td>` +
            `</tr>` +
        `</table>`
    );

    for (let i = 0; i < object_classes.length; i++) {
        let overlay_color = overlay_appearance["colors"][i];
        let color_id = "overlay_color_" + i;
        let obj_cls = object_classes[i];
        let color_picker = 
        `<div style="width: 20px; text-align: left; display: inline">` +
            `<input style="width: 16px; margin: 3px;" type="color" id="${color_id}" name="${color_id}" value="${overlay_color}">` +
        `</div>`;
        $("#color_table").append(
            `<tr>` +
                `<td>` +
                    `<div style="width: 150px">${obj_cls}</div>` +
                `</td>` +
                `<td>` +
                    color_picker +
                `</td>` +
            `</tr>`
        );
    }
}


function create_navigation_table() {

    $("#navigation_table").empty();

    for (let image_name of cur_image_names) {

        let row_id = image_name + "_row";
        let item = 
        `<tr id="${row_id}">` +
            `<td>` +
                `<div class="button-black button-black-hover" style="width: 250px; text-align: left; font-size: 12px; padding: 10px" ` +
                    `onclick="change_image('${image_name}')">` +
                    image_name +
                `</div>` +
            `</td>` +
        `</tr>`;
        $("#navigation_table").append(item);

    }
}




function change_image(image_name) {

    document.getElementById(image_name + "_row").scrollIntoView({behavior: "instant"});

    cur_image_name = image_name;
    let index = cur_image_names.findIndex(x => x == image_name);
    if (index == 0) {
        disable_blue_buttons(["prev_image_button"]);
    }
    else {
        enable_blue_buttons(["prev_image_button"]);
    }
    if (index == cur_image_names.length - 1) {
        disable_blue_buttons(["next_image_button"]);
    }
    else {
        enable_blue_buttons(["next_image_button"]);
    }

    $("#image_name").text(image_name);

    show_image_predictions();
}

function update_fusarium_percentage() {
    let num_fus = 0;
    let num_tot = 0;
    for (let i = 0; i < predictions[cur_image_name]["detector_scores"].length; i++) {
        if (predictions[cur_image_name]["detector_scores"][i] > 0.5) {
            let obj_cls = predictions[cur_image_name]["classes"][i];
            if ((obj_cls == 2) || (obj_cls == 3)) {
                num_fus++;
            }
            if (obj_cls != 0) {
                num_tot++;
            }
        }
    }
    let fus_frac = (num_fus / num_tot);
    let fus_perc = (fus_frac * 100).toFixed(2) + "%";
    $("#fusarium_percentage").html(fus_perc);
}

function show_image_predictions() {

    let pred_callback = function() {

        update_fusarium_percentage();

        if ($("#count_chart").is(":empty")) {
            create_count_chart();
        }
        else {
            update_count_chart();
        }

        let image_set_name = $("#image_set_combo").val();
        let dzi_images_dir = "usr/data/" + username + "/image_sets/" + image_set_name + "/dzi_images";
        let dzi_image_path = ff_path + dzi_images_dir + "/" + cur_image_name + ".dzi";

        viewer.open(dzi_image_path); 
    };

    if (cur_image_name in predictions) {
        pred_callback();
    }
    else {
        img_nav_enabled = false;
        $.post($(location).attr('href'),
        {
            action: "retrieve_predictions",
            image_set_name: $("#image_set_combo").val(),
            image_name: cur_image_name
        },
        function(response, status) {
    
            if (response.error) {
                img_nav_enabled = true;
                show_modal_message("Error", response.message);
            }
            else {
                predictions[cur_image_name] = response.predictions;
                img_nav_enabled = true;
                pred_callback();
            }
        });
    }
}




function change_to_prev_image() {
    let index = cur_image_names.findIndex(x => x == cur_image_name) - 1;
    if (index > -1 && img_nav_enabled) {
        change_image(cur_image_names[index]);
    }
}
function change_to_next_image() {
    let index = cur_image_names.findIndex(x => x == cur_image_name) + 1;
    if (index < cur_image_names.length && img_nav_enabled) {
        change_image(cur_image_names[index]);
    }
}


function populate_image_set_combo() {
    image_set_names = natsort(image_set_names);
    $("#image_set_combo").empty();

    $("#image_set_combo").select2();
    for (let image_set_name of image_set_names) {
        $("#image_set_combo").append($('<option>', {
            value: image_set_name,
            text: image_set_name
        }));

    }
    $("#image_set_combo").prop("selectedIndex", -1);
}




function initialize_class_select() {

    $("#pred_class_select").append($('<option>', {
        value: -1,
        text: "All Classes"
    }));


    let i = 1;

    for (let object_class of object_classes) {
        $("#pred_class_select").append($('<option>', {
            value: i-1,
            text: object_class
        }));
        i++;
    }
    $("#pred_class_select").prop("selectedIndex", 0);

}


$(document).ready(function() {

    image_set_names = data["image_set_names"];
    overlay_appearance = data["overlay_appearance"];

    if (data["maintenance_message"] !== "") {
        $("#maintenance_message").html(data["maintenance_message"]);
        $("#maintenance_message").show();
    }


    populate_image_set_combo();
    initialize_class_select();
    $("#destroy_image_set_button").hide();
    $("#download_csv").hide();
    $("#image_set_buttons_container").show();

    let socket = io(
        "", {
           path: ff_path + "socket.io"
    });

    socket.emit("join_home", username);

    socket.on("upload_change", function(message) {

        let affected_image_set = message["image_set_name"];
        if ($("#image_set_combo").val() === affected_image_set) {
            $("#image_set_combo").val(affected_image_set).change();
        }
    });


    socket.on("progress_change", function(message) {
        let affected_image_set = message["image_set_name"];
        if ($("#image_set_combo").val() === affected_image_set) {
            let progress_message = message["progress"];
            $("#progress_message").html(progress_message);
        }
    });



    $("#image_set_combo").change(function() {

        let image_set_name = $("#image_set_combo").val();
        
        
        $("#pred_class_select").prop("selectedIndex", 0);
        $("#destroy_image_set_button").hide();
        $("#download_csv").hide();
        $("#image_set_area").hide();
        $("#image_set_message").hide();
        predictions = {};
        cur_image_names = [];
        cur_image_name = null;

        $.post($(location).attr('href'),
        {
            action: "fetch_image_set",
            image_set_name: image_set_name
        },
        
        function(response, status) {
            console.log(response);
            if (response.error) {  
                show_modal_message("Error", response.message);  
            }
            else {
                if (response.image_set_status === "processing") {
                    $("#image_set_message").empty();
                    $("#image_set_message").append(
                        `<hr style="margin: 0px">` +
                        `<div style="height: 20px"></div>` +
                        `<div style="font-size: 20px"><i class="fa-regular fa-hourglass-half"></i><span style="margin-left: 10px">Processing</span></div>` +
                        `<div style="height: 10px"></div>` +
                        `<div>This image set is currently being processed. The page will automatically update when the image set is ready to be viewed.</div>` +
                        `<div style="height: 40px"></div>` +
                        `<hr style="margin: auto; width: 200px">` +
                        `<div style="height: 15px"></div>` +
                        `<div id="progress_message" style="width: 300px; display: inline-block; text-align: center; margin-left: 10px; color: white; font-style: italic;">` + response.progress + `</div>` +
                        `<div style="height: 20px"></div>`
                    );
                    $("#image_set_message").show();
                }
                else if (response.image_set_status === "failed") {
                    $("#image_set_message").empty();
                    $("#image_set_message").append(
                        `<hr style="margin: 0px">` +
                        `<div style="height: 20px"></div>` +
                        `<div style="font-size: 20px"><i class="fa-solid fa-face-frown"></i><span style="margin-left: 10px">Failed</span></div>` +
                        `<div style="height: 10px"></div>` +
                        `<div>An error occurred while processing this image set.</div>` +
                        `<div style="height: 20px"></div>`
                    );
                    $("#image_set_message").show();
                    $("#destroy_image_set_button").show();
                    $("#download_csv").hide();
                }
                else if (response.image_set_status === "uploaded") {
                    cur_image_names = response.image_names;

                    let dzi_image_paths = [];
                    let dzi_images_dir = "usr/data/" + username + "/image_sets/" + image_set_name + "/dzi_images";
                    for (let image_name of cur_image_names) {
                        let dzi_image_path = ff_path + dzi_images_dir + "/" + image_name + ".dzi";
                        dzi_image_paths.push(dzi_image_path);
                    }

                    let result_download_path = ff_path + "usr/data/" + username + "/image_sets/" + image_set_name + "/model/result/result.csv";
                    $("#download_csv").attr("href", result_download_path);

                    $("#destroy_image_set_button").show();
                    $("#download_csv").show();
                    $("#image_set_area").show();
                    resize_window();


                    $("#count_chart").empty();
                    create_viewer(dzi_image_paths);
                    create_navigation_table();
                    

                    change_image(cur_image_names[0]);
                }
                else {
                    $("#image_set_combo").prop("selectedIndex", -1);
                    show_modal_message("Error", "Got unexpected image set status: '" + response.image_set_status + "'");
                }
            }
        });
    });

    $("#upload_button").click(function() {
        show_upload_modal();
    });



    $("#next_image_button").click(function() {
        change_to_next_image();
    });

    $("#prev_image_button").click(function() {
        change_to_prev_image();
    });

    $("#image_visible_switch").change(function() {
        viewer.raiseEvent('update-viewport');
    });

    $("#scores_switch").change(function() {
        viewer.raiseEvent('update-viewport');
    });


    $("#pred_class_select").change(function() {
        viewer.raiseEvent('update-viewport');
    });

    $("#destroy_image_set_button").click(function() {
        delete_image_set();
    });

    resize_window();

    $(window).resize(function() {
        resize_window();
    });



    $("body").keydown(function(e) {
        let dropdown_ids = ["image_set_combo", "pred_class_select"];
        if (dropdown_ids.includes(e.target.id)) {
            e.preventDefault();
        }
        // $('#image_set_combo').bind('keypress', function(e) {
        //     e.preventDefault(); 
        // });
    
        // $('#pred_class_select').bind('keypress', function(e) {
        //     e.preventDefault(); 
        // });


        // e.preventDefault();
        keydown_handler(e);
    });
});


function keydown_handler(e) {
    let prev_image_keys = ["ArrowLeft", "ArrowUp"];
    let next_image_keys = ["ArrowRight", "ArrowDown"];
    if (prev_image_keys.includes(e.key)) {
        change_to_prev_image();
    }
    else if (next_image_keys.includes(e.key)) {
        change_to_next_image();
    }
}

function delete_image_set() {


    show_modal_message(`Are you sure?`, 
        `<div id="confirm_message" style="height: 30px">Are you sure you want to delete this image set?</div>` +
        `<div style="height: 10px"></div>` +
        `<div style="text-align: center; height: 40px">` +
            `<div id="confirmation_button_container">` +
                `<button id="confirm_delete_button" class="button-red button-red-hover" style="width: 150px" onclick="confirmed_delete_image_set()">Delete</button>` +
                `<div style="display: inline-block; width: 10px"></div>` +
                `<button id="confirm_close_button" class="button-blue button-blue-hover" style="width: 150px" onclick="close_modal()">Cancel</button>` +
            `</div>` +
            `<div id="loader_container" hidden>` +
                `<div class="loader"></div>` +
            `</div>` +
        `</div>`
    );
}


function confirmed_delete_image_set() {

    $("#modal_close").hide();
    $("#confirmation_button_container").hide();
    $("#modal_head_text").html("Please Wait");
    $("#confirm_message").html("");
    $("#loader_container").show();

    let image_set_to_del = $("#image_set_combo").val();

    $.post($(location).attr('href'),
    {
        action: "delete_image_set",
        image_set_name: image_set_to_del
    },
    function(response, status) {
        close_modal();

        if (response.error) {
            show_modal_message(`Error`, `An error occurred while deleting the result.`);
        }
        else {
            let del_index = image_set_names.indexOf(image_set_to_del);
            if (del_index > -1) {
                image_set_names.splice(del_index, 1);
            }

            populate_image_set_combo();
            $("#destroy_image_set_button").hide();
            $("#download_csv").hide();
            $("#image_set_area").hide();
            $("#image_set_message").hide();
        }
    });
}