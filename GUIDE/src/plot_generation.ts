/**
 * @author Clément Ligneul <clement.ligneul@etu.unistra.fr>
 */

import fs from 'fs';
import { run_obj } from '.';
import * as paths from './paths';

export function parser(data: string, rank:number): number[][] {
    var str = data.toString()
    // parse rows
    var lines = str.split('\n'); //(/(\r?\n)/g);

    var val_col: number[][] = [];

    for (var i = 0; i < lines.length; i++)
        val_col.push([]);

    for (var i = 0; i < lines.length; i++) {
        // parse columns
        var col = lines[i].split('\t');

        if (col.length === 8) {
            for (var j = 0; j < 8; j++) {
                // time
                if (col[j].charAt(col[j].length - 1) === 's')
                    col[j] = col[j].substring(0, col[j].length - 1);

                if (!isNaN(Number(col[j]))) {
                    val_col[i].push(Number(col[j]));
                    if (j !== 7) {
                        write_in_file(Number(col[j]) + ',');
                    } else {
                        write_in_file(Number(col[j]) + ',' + rank +'\n');
                    }
                    
                    // get best fitness generation 
                    // let best_fit_val = Number(run_obj.option_obj.best_fitness_thresh); 
                    // if((Number(col[4]) <= best_fit_val) && (run_obj.best_fit_result_thresh[rank][0] === undefined)){
                    //     run_obj.best_fit_result_thresh[rank][0] = Number(col[0]);
                    //     run_obj.best_fit_result_thresh[rank][1] = Number(col[4]);
                    // }

                    // // get best fitness generation 
                    // let worst_fit_val = Number(run_obj.option_obj.worst_fitness_thresh); 
                    // if((Number(col[4]) <= worst_fit_val) && (run_obj.worst_fit_result_thresh[rank][0] === undefined)){
                    //     run_obj.worst_fit_result_thresh[rank][0] = Number(col[0]);
                    //     run_obj.worst_fit_result_thresh[rank][1] = Number(col[4]);
                    // }

                    // // get best fitness generation 
                    // let avg_fit_val = Number(run_obj.option_obj.avg_fitness_thresh); 
                    // if((Number(col[4]) <= avg_fit_val) && (run_obj.avg_fit_result_thresh[rank][0] === undefined)){
                    //     run_obj.avg_fit_result_thresh[rank][0] = Number(col[0]);
                    //     run_obj.avg_fit_result_thresh[rank][1] = Number(col[4]);
                    // }

                    // // get best fitness generation 
                    // let std_dev_val = Number(run_obj.option_obj.std_dev_thresh); 
                    // if((Number(col[4]) <= std_dev_val) && (run_obj.std_dev_result_thresh[rank][0] === undefined)){
                    //     run_obj.std_dev_result_thresh[rank][0] = Number(col[0]);
                    //     run_obj.std_dev_result_thresh[rank][1] = Number(col[4]);
                    // }

                }
            }
        }
    }
    return val_col;
}

// debug
export function print_data(array: number[][]) {
    for (var i = 0; i < array.length; i++) {
        if (array[i].length === 8) {
            console.log('Generation              : ' + array[i][0]);
            console.log('Time                    : ' + array[i][1]);
            console.log('Planned evaluation      : ' + array[i][2]);
            console.log('Actual evaluation       : ' + array[i][3]);
            console.log('Best individual fitness : ' + array[i][4]);
            console.log('Avg fitness             : ' + array[i][5]);
            console.log('Worst fitness           : ' + array[i][6]);
            console.log('Stand dev               : ' + array[i][7] + '\n');
        }
    }
}

export function write_in_file(buffer: string, file?: string, id?: string, run?:number) {
    let fd = -1;
    let fd2 = -1;
    try {
        if(file){
            fd2 = fs.openSync(file, 'ax');
        } else {
            fd = fs.openSync(run_obj.dir_path + paths.dir_tmp_path + 'data.csv', 'ax');
        }
    } catch {
        if(file){
            fd2 = fs.openSync(file, 'a');
            fs.writeFileSync(fd2, buffer);
            fs.closeSync(fd2);
        } else {
            fd = fs.openSync(run_obj.dir_path + paths.dir_tmp_path + 'data.csv', 'a');
            fs.writeFileSync(fd, buffer);
            fs.closeSync(fd);
        }
        return;
    }

    let init_buf = 'GEN,TIME,PLAN_EVAL,ACTU_EVAL,BEST_FIT,AVG_FIT,WORST_FIT,STD_DEV,RUN';
    if(!file){
        fs.writeFileSync(fd, init_buf + '\n' + buffer);
        fs.closeSync(fd);    
    } else {
        if(id){
            init_buf = `Batch id : ${id}\nBatch size: ${run_obj.batch_size}\nRun_num : ${run}`;
            fs.writeFileSync(fd2, init_buf + '\n' + buffer);
            fs.closeSync(fd2);
        }
    }
}