
const socket_io = require('socket.io');
const io = socket_io({
    "path": process.env.FF_PATH + "socket.io"
});

let home_id_to_key = {};




io.on('connection', function(socket) {
	console.log('A user connected');

    socket.on("join_home", (username) => {
        console.log("join_home from", username);

        home_id_to_key[socket.id] = username;

        console.log("updated home_id_to_key", home_id_to_key);
    });


    socket.on("disconnect", (reason) => {

        if (socket.id in home_id_to_key) {
            console.log("user disconnected from home");

            delete home_id_to_key[socket.id];

            console.log("updated home_id_to_key", home_id_to_key);
        }
    });
});


exports.post_upload_notification = function(req, res, next) {
    let username = req.body.username;
    let image_set_name = req.body.image_set_name;

    console.log("upload update occurred, sending to sockets");
    console.log(username, image_set_name);
    console.log("home_id_to_key", home_id_to_key);

    let key = username;

    for (let socket_id of Object.keys(home_id_to_key)) {
        if (home_id_to_key[socket_id] === key) {
            io.to(socket_id).emit("upload_change", {"image_set_name": image_set_name});
        }
    }

    let response = {};
    response.message = "received";
    return res.json(response);
}


module.exports.io = io;