/**
 * @author Clément Ligneul <clement.ligneul@etu.unistra.fr>
 */

import { ChildProcess } from "child_process";
import { Win_alert } from "./win_alert";
import fs, { appendFileSync, existsSync, readFileSync, writeFileSync } from 'fs';
import { QKeySequence, QShortcut, QWidget } from "@nodegui/nodegui";
import { run_obj } from ".";
import { exit } from "process";

// sleep function equivalent (async call) 
export function sleep(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// return file name from his path
export function get_file_name(path: string): string {
    if (path === '')
        return '';

    var result = '';

    for (var i = 0; i < path.length; i++) {
        var last = path.charAt(path.length - 1);

        if (last === '/') {
            return result;
        } else {
            path = path.substring(0, path.length - 1);
            result = last.concat(result);
        }
    }

    return 'error';
}

// return directory path
export function get_path(str: string): string {
    var res = str;

    for (var i = 0; i < str.length; i++) {
        var last = res.charAt(res.length - 1);

        if (last === '/') {
            return res;
        } else {
            res = res.substring(0, res.length - 1);
        }
    }

    return res;
}

// kill all child processes
export function kill_all(child_array: ChildProcess[]) {
    for (var i = 0; i < child_array.length; i++) {
        if (child_array[i].kill())
            console.log('send kill to process ' + i);

        if (child_array[i].killed) {
            console.log(i + ' is killed');
        } else {
            console.log('error : child not killed : ' + i);
        }
    }
}


export function print_errors(errors: string[]) {
    var message = 'The following parameters are not valid :\n';

    for (var i = 0; i < errors.length; i++)
        message = message.concat('\n- ' + errors[i]);
    
    message.concat('\n');

    new Win_alert(message, 'Parameters Error');
}

// return the best fitness from results
export function get_best_fitness(text: string): number {
    var res: number = 0;
    var row = text.split('\n');

    for (var i = 0; i < row.length; i++) {
        if (row[i].startsWith('EASEA LOG [INFO]: Best fitness:')) {
            res = Number(row[i].split(':')[2]);
        }
    }

    return res
}

// errors correction in csv file
export function fix_csv() {
    var updated = 0;
    var data = fs.readFileSync('/tmp/plotting/data.csv', 'utf8');

    var rows = data.split('\n');
    var index = rows[0].length + 1; // position in the text

    for (var i = 1; i < rows.length; i++) {
        var sub_str = rows[i].split(',');

        if (sub_str.length > 8) {
            updated = 1;
            var p = 0;
            var id = 0;

            while (id != rows[i].length) {
                if (rows[i][id] === ',' && p === 7) {
                    rows[i] = rows[i].substr(0, id) + '\n' + rows[i].substr(id + 1);
                    console.log('new row = ' + rows[i])
                    // break;
                } else if (rows[i][id] === ',') {
                    p++;
                }
                id++;
            }
        }
    }
    if (updated) {
        rows[0] += '\n';
        var t = '';
        for (var j = 0; j < rows.length; j++) {
            // rows[j] = rows[j].substr(1);
            rows[j] += '\n';
            t += rows[j];
        }

        fs.unlinkSync('/tmp/plotting/data.csv');
        fs.writeFileSync('/tmp/plotting/data.csv', t);
    }
}

// used to disable escape key (can be used for a shortcut)
export function disable_keys(keys: string, window: QWidget) {
    const shortcut = new QShortcut(window);
    shortcut.setKey(new QKeySequence(keys));
    shortcut.setEnabled(true);
    shortcut.addEventListener("activated", () => { });
}

// return the ip:port list from ip.txt
export function read_ip_file(path:string):string[]
{
    var ip_list:string[] = [];
    if(!existsSync(path))
        console.log(`Error : ${path} not found`);
    
    var data = readFileSync(path, 'utf-8');
    data.split('\n').forEach(function(val, index, array){
        var reg = new RegExp("^(?:[0-9]{1,3}\.){3}[0-9]{1,3}\:[0-9]{4}");

        if(index < array.length-1){
            if(val.trim() != "" && reg.test(val)){
                ip_list.push(val);
            } else {
                console.log(`${path} : "${val}" is not valid`);
            }
        }
    });

    return ip_list;
}

/**
 * @brief Find the value associated with a parameter
 * @param file The ez file
 * @param parameter The parameter to find
 * @param pressure Return pressure
 * @returns Empty string if not found else return the value associated
 */
export function parse_ez_file(file:string, parameter:string, pressure?: boolean): string
{
    var res = "";
    var file = readFileSync(file, "utf-8");

    file.split('\n').forEach((line)=>{
        if(new RegExp(parameter + " *:.*", "g").test(line)){
            var text = line.split(':');
            res = text[1];

            if(text[1].search('//') != -1){
                res = text[1].split("//")[0].trim();

                if(res.search('/*') != -1)
                    res = res.split('/*')[0].trim();
            } 
            
            if(text[1].search('/*') != -1){
                res = text[1].split('/*')[0].trim();

                if(res.search('//') != -1)
                    res = res.split('//')[0].trim();
            }

            if(pressure){
                var m = res.match(/(\d*[.])?\d+/);

                if(m)
                    res = m[0];
            }
        }
    });

    return res;
}

export function write_log(path:string, rank:number, data:string) 
{
    rank -= 1;
    var ez_file = run_obj.ez_file_address;
    
    if(!existsSync(path)){
        var init_text = `Run configuration :\n\n`;
        init_text += `Start time : ${new Date(run_obj.batch_id)}\n`;
        init_text += `Seed : ${run_obj.option_obj.proc_tab[rank].seed_value ? run_obj.option_obj.proc_tab[rank].seed_value : run_obj.option_obj.seed ? run_obj.option_obj.seed : (run_obj.batch_id + rank)}\n`;
        init_text += `Number of generations : ${run_obj.option_obj.nb_gen ? run_obj.option_obj.nb_gen : parse_ez_file(ez_file, "Number of generations")}\n`;
        init_text += `Population size : ${run_obj.option_obj.pop_size ? run_obj.option_obj.pop_size : parse_ez_file(ez_file, "Population size")}\n`;
        init_text += `CPU threads number : ${run_obj.option_obj.thread_number ? run_obj.option_obj.thread_number : 20/*parse_ez_file(ez_file, "CPU threads number")*/}\n`;
        init_text += '____________________________________________________\n';
        init_text += '\nBatch information : \n\n';
        init_text += `Batch size : ${run_obj.batch_size}\n`;
        init_text += `Batch ID : ${run_obj.batch_id}\n`;
        if(run_obj.island_model){
            init_text += `Run number : ${Math.floor(run_obj.batch_size - rank/run_obj.island_obj.nb_isl_per_run) + 1}/${run_obj.batch_size}\n`;
        } else {
            init_text += `Run number : ${rank + 1}/${run_obj.batch_size}\n`;

        }

        init_text += '____________________________________________________\n';
        init_text += '\nSpecial options :\n\n';
        init_text += `Offspring population size : ${!isNaN(run_obj.off_obj.size_off) ? run_obj.off_obj.size_off : parse_ez_file(ez_file, "Offspring size")}\n`;         
        init_text += `Mutation probability : ${parse_ez_file(ez_file, "Mutation probability")}\n`;
        init_text += `Crossover probability : ${parse_ez_file(ez_file, "Crossover probability")}\n`;
        init_text += `Selection operator : ${run_obj.option_obj.select_op}\n`;
        init_text += `Selection pressure : ${run_obj.option_obj.select_pressure ? run_obj.option_obj.select_pressure : parse_ez_file(ez_file, "Selection operator", true)}\n`;
        init_text += `Reduce parents operator : ${run_obj.parent_obj.reduce_op}\n`;
        init_text += `Reduce parent pressure : ${run_obj.parent_obj.reduce_pressure != 2.0 ? run_obj.parent_obj.reduce_pressure : parse_ez_file(ez_file, "Reduce parents operator", true)}\n`;
        init_text += `Reduce offspring operator : ${run_obj.off_obj.reduce_op}\n`;
        init_text += `Reduce offspring pressure : ${run_obj.off_obj.reduce_pressure != 2.0 ? run_obj.off_obj.reduce_pressure : parse_ez_file(ez_file, "Reduce offspring operator", true)}\n`;
        init_text += `Surviving parents : ${run_obj.parent_obj.surviving ? run_obj.parent_obj.surviving : parse_ez_file(ez_file, "Surviving parents")}\n`;
        init_text += `Surviving offspring : ${run_obj.off_obj.surviving_off ? run_obj.off_obj.surviving_off : parse_ez_file(ez_file, "Surviving parents")}\n`;
        // init_text += `Replacement operator: ${run_obj.option_obj.}`
        // init_text += `Replacement pressure: 2
        init_text += `Elitism : ${run_obj.option_obj.elite_type === 0 ? "Weak" : "Strong"}\n`;
        init_text += `Elite size : ${run_obj.option_obj.nb_elite ? run_obj.option_obj.nb_elite : parse_ez_file(ez_file, "Elite")}\n`;
        init_text += '____________________________________________________\n';
        // init_text += '\nIslands model:\n\n';
        init_text += `\nIslands model : ${run_obj.island_model ? run_obj.island_obj.local ? "local islands" : "remote islands" : "disabled"}\n\n`;
        if(run_obj.island_model){
            init_text += `Migration probability : ${run_obj.island_obj.migration_proba}\n`;
            init_text += `Server port : ${2929 + rank}\n`;
            init_text += `Reevaluate immigrants : ${run_obj.island_obj.reevaluate}\n`;
            init_text += `Island number : ${rank%run_obj.island_obj.nb_isl_per_run} rank ${rank}\n`;
        }
        init_text += '____________________________________________________\n';
        init_text += '\nOutput:\n\n';

        writeFileSync(path, init_text, "utf-8");
    }

    appendFileSync(path, data, "utf-8");

}