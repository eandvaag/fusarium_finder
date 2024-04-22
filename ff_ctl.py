#!/usr/bin/python3

import logging
import sys
import argparse
import os
import json
import yaml
import subprocess

def create():

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    args_path = os.path.join(".", "args.json")
    with open(args_path, 'r') as fp:
        args = json.load(fp)


    logger.info("Configuring docker-compose.yml")

    docker_compose_init_path = os.path.join("site", "docker-compose-init.yml")
    with open(docker_compose_init_path, 'r') as ymlfile:
        conf = yaml.safe_load(ymlfile)


    psql_env = conf["services"]["db"]["environment"]
    psql_env["POSTGRES_DB"] = args["postgres_db_name"]
    psql_env["POSTGRES_USER"] = args["postgres_db_username"]
    psql_env["POSTGRES_PASSWORD"] = args["postgres_db_password"]

    conf["services"]["db"]["ports"] = [str(args["postgres_db_port"]) + ":5432"]


    site_env = conf["services"]["myapp"]["environment"]
    site_env["DB_SCHEMA"] = args["postgres_db_name"]
    site_env["DB_USER"] = args["postgres_db_username"]
    site_env["DB_PASSWORD"] = args["postgres_db_password"]

    site_env["FF_PORT"] = args["site_port"]
    site_env["FF_PY_PORT"] = args["backend_python_port"] 
    url_path_prefix = args["url_path_prefix"]
    if url_path_prefix == "":
        url_path_prefix = "/"
    if url_path_prefix[0] != "/":
        url_path_prefix = "/" + url_path_prefix
    if url_path_prefix[-1] != "/":
        url_path_prefix = url_path_prefix + "/"
    site_env["FF_PATH"]  = url_path_prefix
    site_env["FF_API_KEY"] = args["api_key"]
    site_env["FF_GPU_INDEX"] = args["gpu_index"]



    timezone_command = "timedatectl | grep 'Time zone' | awk '{ print $3 }'"
    timezone = (subprocess.check_output(timezone_command, shell=True)).decode("utf-8").strip()
    site_env["FF_TIMEZONE"] = timezone


    conf["services"]["myapp"]["ports"] = [str(args["site_port"]) + ":" + str(args["site_port"])]


    cwd = os.getcwd()
    site_volume = conf["services"]["myapp"]["volumes"][0]
    site_volume["source"] = os.path.join(cwd, "backend", "src", "usr")



    with open("docker-compose.yml", "w") as ymlfile:
        yaml.dump(conf, ymlfile, default_flow_style=False)


    logger.info("Writing seeders file")

    seeders_dir = os.path.join("site", "myapp", "seeders")

    seeders_name = "seed-users.js"
    seeders_path = os.path.join(seeders_dir, seeders_name)

    f = open(seeders_path, "w")
    f.write(
        "'use strict';\n" +
        "\n" +
        "var bcrypt = require('bcrypt');\n" +
        "\n" +
        "module.exports = {\n" + 
        "    up: (queryInterface, Sequelize) => {\n" +
        "\n" +
        "        const salt = bcrypt.genSaltSync();\n" +
        "        return queryInterface.bulkInsert('users', [\n" +
        "            {\n" +
        "                username: '" + args["admin_username"] + "',\n" +
        "                password: bcrypt.hashSync('" + args["admin_password"] + "', salt),\n" +
        "                is_admin: true,\n"
        "                createdAt: new Date(),\n" +
        "                updatedAt: new Date()\n" +
        "            }\n" +
        "        ], {\n" +
        "        });\n" +
        "    },\n" +
        "    down: (queryInterface, Sequelize) => {\n" +
        "        return queryInterface.bulkDelete('users', null, {});\n" +
        "    }\n" +
        "};"
    )



    f.close()


    logger.info("Writing config.js")


    config_path = os.path.join("site", "myapp", "config", "config.js")

    f = open(config_path, "w")
    f.write(
        'module.exports = {\n' +
        '    "docker": {\n' +
        '        "username": "' + args["postgres_db_username"] + '",\n' +
        '        "password": "' + args["postgres_db_password"] + '",\n' +
        '        "database": "' + args["postgres_db_name"] + '",\n' +
        '        "host": "db",\n' +
        '        "dialect": "postgres",\n' +
        '        "port": 5432\n' +
        '    }\n' +
        '}'
    )
    f.close()

    logger.info("Creating symlink for usr dir (if missing)")
    symlink_path = os.path.join("site", "myapp", "usr")
    if not os.path.islink(symlink_path):
        usr_dir = os.path.join("backend", "src", "usr")
        os.symlink(usr_dir, symlink_path)


    logger.info("Starting Docker container")

    subprocess.run(["docker-compose", "up", "-d"])





def up():
    subprocess.run(["docker-compose", "up", "-d"])

def down():
    subprocess.run(["docker-compose", "down", "--rmi", "local"])

def destroy():
    subprocess.run(["docker-compose", "down", "-v", "--rmi", "local"])



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="fusariumfinder",
        description="Control program for running Fusarium Finder in a Docker container"
    )

    parser.add_argument("-c", "--create", action="store_true",
                        help="create the docker container")

    parser.add_argument("-d", "--down", action="store_true",
                        help="stop the docker container")

    parser.add_argument("-u", "--up", action="store_true",
                        help="start the docker container")

    parser.add_argument("-D", "--destroy", action="store_true",
                        help="remove the docker container")
    
    
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        exit(1)

    if len(sys.argv) > 2:
        parser.print_help()
        exit(1)

    if args.create:
        create()
    elif args.down:
        down()
    elif args.up:
        up()
    elif args.destroy:
        destroy()

    exit(0)