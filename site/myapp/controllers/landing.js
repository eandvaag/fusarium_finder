const path = require('path');
const fs = require('fs');
const { glob } = require("glob");


const { Op } = require("sequelize");
const models = require('../models');

var bcrypt = require('bcrypt');

const USR_DATA_ROOT = path.join("usr", "data");
const USR_SHARED_ROOT = path.join("usr", "shared");

let active_uploads = {};

const nat_orderBy = require('natural-orderby');

const { spawn, exec } = require('child_process');


const FILE_FORMAT = /[\s `!@#$%^&*()+\=\[\]{};':"\\|,<>\/?~]/;
const IMAGE_SET_NAME_FORMAT = /[\s `!@#$%^&*()+\=\[\]{}.;':"\\|,<>\/?~]/;

const MAX_EXTENSIONLESS_FILENAME_LENGTH = 100;




const USERNAME_FORMAT = /[\s `!@#$%^&*()+\=\[\]{}.;':"\\|,<>\/?~]/;
const MIN_USERNAME_LENGTH = 1;
const MAX_USERNAME_LENGTH = 255;
const MIN_PASSWORD_LENGTH = 1;
const MAX_PASSWORD_LENGTH = 255;


if (process.env.NODE_ENV === "docker") {
    console.log("Starting the python server...");

    let scheduler_args = ["../../backend/src/server.py"];
    let scheduler = spawn("python3", scheduler_args);

    scheduler.on('close', (code) => {
        console.log(`Scheduler closed with code ${code}.`);
    });

    scheduler.on('exit', (code) => {
        console.log(`Scheduler exited with code ${code}.`);
    });

    scheduler.stderr.on('data', (data) => {
        console.error(`Scheduler stderr: ${data}`);
    });
    scheduler.stdout.on('data', (data) => {
        console.log(`Scheduler stdout: ${data}`);
    });

    scheduler.on('SIGINT', function() {
        console.log('Scheduler received SIGINT signal');
    });

    scheduler.on('error', (error) => {
        console.log("Failed to start scheduler subprocess.");
        console.log(error);
    });

}

function get_subdirnames(dir) {
    let subdirnames = [];
    let list = fs.readdirSync(dir);
    list.forEach(function(file) {
        let fpath = path.join(dir, file);
        let stat = fs.statSync(fpath);
        if (stat && stat.isDirectory()) {
            subdirnames.push(file);
        }
    });
    return subdirnames;
}


function fpath_exists(fpath) {
    let exists = true;
    try {
        fs.accessSync(fpath, fs.constants.F_OK);
    }
    catch (e) {
        exists = false;
    }
    return exists;
}


exports.get_sign_in = function(req, res, next) {
    res.render('sign_in');
}


exports.post_sign_in = function(req, res, next) {
    let response = {};
    response.not_found = false;
    response.error = false;

    return models.users.findOne({
    where: {
        username: req.body.username,
    }
    }).then(user => {
        if (!user) {
            response.not_found = true;
            return res.json(response);
        }
        else {
            if (!user.check_password(req.body.password)) {
                response.not_found = true;
                return res.json(response);
            }
            else {
                if (user.is_admin) {
                    req.session.user = user.dataValues;
                    response.redirect = process.env.FF_PATH + "admin";
                    return res.json(response);
                }
                else {
                    console.log("redirecting user to their home dir");
                    req.session.user = user.dataValues;
                    response.redirect = process.env.FF_PATH + "home/" + req.body.username;
                    return res.json(response);
                }
            }
        }
    }).catch(error => {
        console.log(error);
        response.error = true;
        return res.json(response);
    });
}


exports.get_admin = function(req, res, next) {

    if (req.session.user && req.cookies.user_sid) {

        return models.users.findAll({
            where: {
                username: {
                    [Op.not]: "admin"
                }
            }
        }).then(users => {

            let data = {};
            data["users"] = users;
            
            res.render("admin", {
                ff_path: process.env.FF_PATH,
                username: req.session.user.username, 
                data: data
            });

        }).catch(error => {
            console.log(error);
            return res.redirect(process.env.FF_PATH);
        });
    }
    else {
        return res.redirect(process.env.FF_PATH);
    }
}
function init_usr(username) {

    let usr_dirs = [
        path.join(USR_DATA_ROOT, username),
        path.join(USR_DATA_ROOT, username, "image_sets")
    ];


    for (let usr_dir of usr_dirs) {
        console.log("creating directory", usr_dir);
        try {
            fs.mkdirSync(usr_dir, { recursive: true });
        }
        catch (error) {
            console.log(error);
            return false;
        }
    }


    let default_overlay_apperance_path = path.join(USR_SHARED_ROOT, "default_overlay_appearance.json");
    let overlay_appearance_path = path.join(USR_DATA_ROOT, username, "overlay_appearance.json");
    console.log("copying default overlay appearance");
    try {
        fs.copyFileSync(default_overlay_apperance_path, overlay_appearance_path, fs.constants.COPYFILE_EXCL);
    }
    catch (error) {
        console.log(error);
        return false;
    }


    return true;
}

exports.post_admin = function(req, res, next) {


    let action = req.body.action;
    let response = {};

    if (action === "update_user_password") {

        let username = req.body.username;
        let password = req.body.password;


        if ((typeof username !== 'string') && (!(username instanceof String))) {
            response.message = "The provided username is not a string.";
            response.error = true;
            return res.json(response);
        }

        if ((typeof password !== 'string') && (!(password instanceof String))) {
            response.message = "The provided password is not a string.";
            response.error = true;
            return res.json(response);
        }

        if (password.length < MIN_PASSWORD_LENGTH) {
            response.message = "The provided password is too short.";
            response.error = true;
            return res.json(response);
        }

        if (password.length > MAX_PASSWORD_LENGTH) {
            response.message = "The provided password is too long.";
            response.error = true;
            return res.json(response);
        }


        return models.users.findOne({
            where: {
                username: username
            }
        }).then(user => {
            if (!user) {
                response.message = "The user could not be found in the database.";
                response.error = true;
                return res.json(response);
            }
            else {
                const salt = bcrypt.genSaltSync();
                return models.users.update({
                    password: bcrypt.hashSync(req.body.password, salt),
                }, {
                    where: {
                        username: req.body.username
                    }
                }).then(user => {
                    response.error = false;
                    return res.json(response);
                }).catch(error => {
                    console.log(error);
                    response.message = "An error occurred while updating the user password.";
                    response.error = true;
                    return res.json(response);
                });
            }
        }).catch(error => {
            console.log(error);
            response.message = "An error occurred while checking for the user's existence in the database.";
            response.error = true;
            return res.json(response);
        });

    }

    else if (action === "create_user_account") {

        
        let username = req.body.username;
        let password = req.body.password;


        if ((typeof username !== 'string') && (!(username instanceof String))) {
            response.message = "The provided username is not a string.";
            response.error = true;
            return res.json(response);
        }

        if ((typeof password !== 'string') && (!(password instanceof String))) {
            response.message = "The provided password is not a string.";
            response.error = true;
            return res.json(response);
        }

        if (USERNAME_FORMAT.test(username)) {
            response.message = "The provided username contains illegal characters.";
            response.error = true;
            return res.json(response);
        }

        if (username.length < MIN_USERNAME_LENGTH) {
            response.message = "The provided username is too short.";
            response.error = true;
            return res.json(response);
        }

        if (username.length > MAX_USERNAME_LENGTH) {
            response.message = "The provided username is too long.";
            response.error = true;
            return res.json(response);
        }

        if (password.length < MIN_PASSWORD_LENGTH) {
            response.message = "The provided password is too short.";
            response.error = true;
            return res.json(response);
        }

        if (password.length > MAX_PASSWORD_LENGTH) {
            response.message = "The provided password is too long.";
            response.error = true;
            return res.json(response);
        }


        return models.users.findOne({
            where: {
                username: username
            }
        }).then(user => {
            if (user) {
                response.message = "The provided username is in use by an existing account.";
                response.error = true;
                return res.json(response);
            }
            else {
                    
                let dirs_initialized = init_usr(username);
                
                if (dirs_initialized) {
                    return models.users.create({
                        username: req.body.username,
                        password: req.body.password,
                        is_admin: false
                    }).then(user => {

                        response.error = false;
                        return res.json(response);
                    }).catch(error => {
                        console.log(error);
                        let usr_dir = path.join(USR_DATA_ROOT, username);
                        try {
                            fs.rmSync(usr_dir, { recursive: true, force: false });
                        }
                        catch(error) {
                            console.log(error);
                        }
                        response.message = "An error occurred while creating the user account.";
                        response.error = true;
                        return res.json(response);
                    });

                }
                else {
                    response.message = "An error occurred while initializing the user's directory tree.";
                    response.error = true;
                    return res.json(response);
                }

            }
        }).catch(error => {
            console.log(error);
            response.message = "An error occurred while checking for user account uniqueness.";
            response.error = true;
            return res.json(response);
        });

    }
}


exports.get_home = function(req, res, next) {

    console.log("get home");

    if ((req.session.user && req.cookies.user_sid) && (req.params.username === req.session.user.username)) {

        let username = req.session.user.username;

        console.log("getting image sets");
        let image_sets_root = path.join(USR_DATA_ROOT, username, "image_sets");
        let image_set_names;
        try {
           image_set_names = get_subdirnames(image_sets_root);
        }
        catch (error) {
            return res.redirect(process.env.FF_PATH);
        }

        let overlay_appearance;
        let overlay_appearance_path = path.join(USR_DATA_ROOT, username, "overlay_appearance.json");
        try {
            overlay_appearance = JSON.parse(fs.readFileSync(overlay_appearance_path, 'utf8'));
        }
        catch (error) {
            console.log(error);
            return res.redirect(process.env.FF_PATH);
        }


        let maintenance_message = "";
        let maintenance_path = path.join(USR_SHARED_ROOT, "maintenance.json");
        if (fpath_exists(maintenance_path)) {
            try {
                maintenance_log = JSON.parse(fs.readFileSync(maintenance_path, 'utf8'));
            }
            catch (error) {
                console.log(error);
                return res.redirect(process.env.AC_PATH);
            }

            maintenance_message = maintenance_log["message"];
        }


        let data = {};
        data["image_set_names"] = image_set_names;
        data["overlay_appearance"] = overlay_appearance;
        data["maintenance_message"] = maintenance_message;

        console.log("rendering home page");
    
        res.render("home", {
            ff_path: process.env.FF_PATH,
            username: username, 
            data: data
        });

    }
    else {
        return res.redirect(process.env.FF_PATH);
    }

}


exports.post_home = async function(req, res, next) {

    let action = req.body.action;
    let username = req.session.user.username;
    let response = {};

    if (action === "fetch_image_set") {
        console.log("fetch_image_set", req.body.image_set_name)
        let image_set_name = req.body.image_set_name;

        let image_set_dir = path.join(USR_DATA_ROOT, username, "image_sets", image_set_name);
        console.log("image_set_dir", image_set_dir);
        let status_path = path.join(image_set_dir, "upload_status.json");
        let status;
        try {
            status = JSON.parse(fs.readFileSync(status_path, 'utf8'));
        }
        catch (error) {
            response.error = true;
            response.message = "Failed to read image set status.";
            return res.json(response);
        }
        console.log("status", status);

        response.image_set_status = status["status"];
        if (status["status"] === "uploaded") {


            let images_dir = path.join(image_set_dir, "images");
            console.log(images_dir);
            let image_paths;
            try {
                image_paths = await glob(path.join(images_dir, "*"));
            }
            catch (error) {
                console.log(error);
                response.error = true;
                response.message = "Failed to retrieve image listing.";
                return res.json(response);
            }
            let image_names = [];
            for (let image_path of image_paths) {
                let image_name_with_ext = path.basename(image_path);
                let image_name = image_name_with_ext.split(".")[0];
                image_names.push(image_name);
            }
            console.log(image_names);
            image_names = nat_orderBy.orderBy(image_names);
            console.log(image_names);
            response.image_names = image_names;
            response.error = false;
            return res.json(response);

            // glob(path.join(images_dir, "*"), function(error, image_paths) {
            //     if (error) {
            //         console.log(error);
            //         response.error = true;
            //         response.message = "Failed to retrieve image listing.";
            //         return res.json(response);
            //     }
            //     console.log("no error");
            //     let image_names = [];
            //     for (let image_path of image_paths) {
            //         let image_name_with_ext = path.basename(image_path);
            //         let image_name = image_name_with_ext.split(".")[0];
            //         image_names.push(image_name);
            //     }
            //     console.log(image_names);
            //     image_names = nat_orderBy.orderBy(image_names);
            //     console.log(image_names);
            //     response.image_names = image_names;
            //     response.error = false;
            //     return res.json(response);
            // });
        }
        else {
            response.error = false;
            return res.json(response);
        }
    }

    else if (action === "retrieve_predictions") {
        let image_set_name = req.body.image_set_name;
        let image_name = req.body.image_name;

        let image_set_dir = path.join(USR_DATA_ROOT, username, "image_sets", image_set_name);

        let predictions_path = path.join(image_set_dir, "model", "result", "prediction", image_name + ".json");

        let predictions;
        try {
            predictions = JSON.parse(fs.readFileSync(predictions_path, 'utf8'));
        }
        catch (error) {
            response.error = true;
            response.message = "Failed to retrieve predictions.";
            return res.json(response);
        }

        response.error = false;
        response.predictions = predictions;
        return res.json(response);
    }

    else if (action === "delete_image_set") {
        let image_set_name = req.body.image_set_name;
        let image_set_dir = path.join(USR_DATA_ROOT, username, "image_sets", image_set_name);

        let status_path = path.join(image_set_dir, "upload_status.json");
        let status;
        try {
            status = JSON.parse(fs.readFileSync(status_path, 'utf8'));
        }
        catch (error) {
            response.error = true;
            response.message = "Failed to read image set status.";
            return res.json(response);
        }

        response.image_set_status = status["status"];
        if (status["status"] !== "uploaded" && status["status"] !== "failed") {
            response.error = true;
            response.message = "Incorrect image set status.";
            return res.json(response);
        }
        try {
            remove_image_set(username, image_set_name);
        }
        catch (error) {
            response.error = true;
            response.message = "Failed to delete image set.";
            return res.json(response);
        }

        response.error = false;
        return res.json(response);
    }

    else if (action === "save_overlay_appearance") {
        let overlay_appearance = JSON.parse(req.body.overlay_appearance);

        if (Object.keys(overlay_appearance).length != 1) {
            response.error = true;
            response.message = "Invalid overlay appearance.";
            return res.json(response);
        }

        if (!("colors" in overlay_appearance)) {
            response.error = true;
            response.message = "Invalid overlay appearance.";
            return res.json(response);
        }

        let overlay_appearance_path = path.join(USR_DATA_ROOT, username, "overlay_appearance.json");
        try {
            fs.writeFileSync(overlay_appearance_path, JSON.stringify(overlay_appearance));
        }
        catch (error) {
            response.error = true;
            response.message = "Failed to save overlay appearance.";
            return res.json(response);
        }

        response.error = false;
        return res.json(response);
    }
}

function remove_image_set(username, image_set_name) {

    console.log("remove_image_set");
    console.log("username", username);
    console.log("image_set_name", image_set_name);

    if ((username === "" || image_set_name === "")) {
        throw "Empty string argument provided";
    }

    if ((username == null || image_set_name == null)) {
        throw "Null argument provided";
    }

    let image_set_dir = path.join(USR_DATA_ROOT, username, "image_sets", image_set_name);

    if (fs.existsSync(image_set_dir)) {
        console.log("removing image_set_dir", image_set_dir);
        fs.rmSync(image_set_dir, { recursive: true, force: false });
    }
}




exports.post_image_set_upload = async function(req, res, next) {

    let upload_uuid;
    let image_set_name;
    let first;
    let last;
    let queued_filenames;

    let username = req.session.user.username;
    
    if (req.files.length > 1) {
        upload_uuid = req.body.upload_uuid[0];
        image_set_name = req.body.image_set_name[0];
        first = false;
        last = false;
        queued_filenames = req.body.queued_filenames[0].split(",");
        let num_sent;
        for (let i = 0; i < req.body.num_sent.length; i++) {
            num_sent = parseInt(req.body.num_sent[i]);
            if (num_sent == 1) {
                first = true;
            }
            if (num_sent == queued_filenames.length) {
                last = true;
            }
        }
        
    }
    else {
        upload_uuid = req.body.upload_uuid;
        image_set_name = req.body.image_set_name;
        queued_filenames = req.body.queued_filenames.split(",");
        first = parseInt(req.body.num_sent) == 1;
        last = parseInt(req.body.num_sent) == queued_filenames.length;
    }

    console.log("first?", first);
    console.log("last?", last);

    
    if (first) {
        if (upload_uuid in active_uploads) {
            return res.status(422).json({
                error: "Upload key conflict."
            });
        }
        else {
            active_uploads[upload_uuid] = queued_filenames.length;
        }
    }
    else {
        if (!(upload_uuid in active_uploads)) {
            return res.status(422).json({
                error: "Upload is no longer active."
            });
        }
        else {
            if (req.files.length > 1) {
                for (let i = 0; i < req.body.queued_filenames.length; i++) {
                    let queued_filenames = req.body.queued_filenames[i].split(",");
                    if (queued_filenames.length != active_uploads[upload_uuid]) {

                        try {
                            remove_image_set(username, image_set_name);
                        }
                        catch (error) {
                            console.log("Failed to remove image set");
                            console.log(error);
                        }

                        return res.status(422).json({
                            error: "Size of image set changed during upload."
                        });
                    }
                }
            }
            else {
                let queued_filenames = req.body.queued_filenames.split(",");
                if (queued_filenames.length != active_uploads[upload_uuid]) {

                    try {
                        remove_image_set(username, image_set_name);
                    }
                    catch (error) {
                        console.log("Failed to remove image set");
                        console.log(error);
                    }

                    return res.status(422).json({
                        error: "Size of image set changed during upload."
                    });
                }
            }
        }
    }

    let image_sets_root = path.join(USR_DATA_ROOT, username, "image_sets");
    let image_set_dir = path.join(image_sets_root, image_set_name);
    let images_dir = path.join(image_set_dir, "images");

    
    if (first) {

        for (let filename of queued_filenames) {
            if (FILE_FORMAT.test(filename)) {
                delete active_uploads[upload_uuid];
                return res.status(422).json({
                    error: "One or more provided filenames contains illegal characters."
                });
            }
            let split_filename = filename.split(".");
            if ((split_filename.length != 1) && (split_filename.length != 2)) {
                delete active_uploads[upload_uuid];
                return res.status(422).json({
                    error: "At least one filename contains an illegal '.' character."
                });

            }

            let extensionless_fname = split_filename[0];
            if (extensionless_fname.length > MAX_EXTENSIONLESS_FILENAME_LENGTH) {
                delete active_uploads[upload_uuid];
                return res.status(422).json({
                    error: "One or more filenames exceeds maximum allowed length of " + MAX_EXTENSIONLESS_FILENAME_LENGTH + " characters."
                });
            }
    
        }


        if (fpath_exists(image_set_dir)) {
            
            delete active_uploads[upload_uuid];
            return res.status(422).json({
                error: "Image set names must be unique! The provided name is already in use."
            });
        }
        if (image_set_name.length < 3) {
            delete active_uploads[upload_uuid];
            return res.status(422).json({
                error: "The provided image set name is too short."
            });
        }
        if (image_set_name.length > 20) {
            delete active_uploads[upload_uuid];
            return res.status(422).json({
                error: "The provided image set name is too long."
            });
        }
        if (IMAGE_SET_NAME_FORMAT.test(image_set_name)) {
            delete active_uploads[upload_uuid];
            return res.status(422).json({
                error: "The provided image set name contains illegal characters."
            });
        }


        console.log("Making the images directory");
        fs.mkdirSync(images_dir, { recursive: true });

    }
    else {
        if (!(fpath_exists(image_set_dir))) {
            try {
                remove_image_set(username, image_set_name);
            }
            catch (error) {
                console.log("Failed to remove image set");
                console.log(error);
            }
            delete active_uploads[upload_uuid];
            return res.status(422).json({
                error: "Image set directories were not created by initial request."
            });
        }
    }

    console.log("Writing the image files");
    for (let file_index = 0; file_index < req.files.length; file_index++) {
        let file = req.files[file_index];
        console.log(file);
        console.log(file.buffer);

        let split_filename = file.originalname.split(".");
        let extensionless_fname = split_filename[0];
        if (extensionless_fname.length > MAX_EXTENSIONLESS_FILENAME_LENGTH) {
            try {
                remove_image_set(username, image_set_name);
            }
            catch (error) {
                console.log("Failed to remove image set");
                console.log(error);
            }
            delete active_uploads[upload_uuid];
            return res.status(422).json({
                error: "One or more filenames exceeds maximum allowed length of " + MAX_EXTENSIONLESS_FILENAME_LENGTH + " characters."
            });
        }

        let fpath = path.join(images_dir, file.originalname);
        try {
            fs.writeFileSync(fpath, file.buffer);
        }
        catch (error) {
            console.log(error);
            try {
                remove_image_set(username, image_set_name);
            }
            catch (error) {
                console.log("Failed to remove image set");
                console.log(error);
            }
            delete active_uploads[upload_uuid];
            return res.status(422).json({
                error: "Error occurred when writing image file."
            });
        }
    }

    if (last) {

        let config = {
            "username": username,
            "image_set_name": image_set_name,
        }


        let config_path = path.join(image_set_dir, "config.json");
        try {
            fs.writeFileSync(config_path, JSON.stringify(config));
        }
        catch (error) {
            delete active_uploads[upload_uuid];
            return res.status(422).json({
                error: "Error occurred when writing configuration file."
            });
        }


        let upload_status_path = path.join(image_set_dir, "upload_status.json")
        try {
            fs.writeFileSync(upload_status_path, JSON.stringify({"status": "processing"}));
        }
        catch (error) {
            delete active_uploads[upload_uuid];
            return res.status(422).json({
                error: "Error occurred when writing upload file."
            });
        }

        let process_upload_command = "python ../../backend/src/process_upload.py " + image_set_dir;
        exec(process_upload_command, {shell: "/bin/bash"}, function (error, stdout, stderr) {
            if (error) {
                console.log(error.stack);
                console.log('Error code: '+error.code);
                console.log('Signal received: '+error.signal);
            }
        });

        delete active_uploads[upload_uuid];

    }

    return res.sendStatus(200);

}



exports.logout = function(req, res, next) {
    console.log("logging out");
    if (req.session.user && req.cookies.user_sid) {
        console.log("clearing cookies");
        res.clearCookie('user_sid');
        console.log("cookies cleared");
    }
    console.log("redirecting");
    return res.redirect(process.env.FF_PATH);
}