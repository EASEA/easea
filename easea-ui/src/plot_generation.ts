/**
 * @author Clément Ligneul <clement.ligneul@etu.unistra.fr>
 */

import fs from 'fs';

export const data_csv = '/tmp/plotting/data.csv';

export function parser(data: string): number[][] {
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
                        write_in_file(Number(col[j]) + '\n');
                    }
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

export function write_in_file(buffer: string) {
    var fd: number;
    try {
        fd = fs.openSync(data_csv, 'ax');
    } catch {
        fd = fs.openSync(data_csv, 'a');
        fs.writeFileSync(fd, buffer);
        fs.closeSync(fd);
        return;
    }

    var init_buf = 'GEN,TIME,PLAN_EVAL,ACTU_EVAL,BEST_FIT,AVG_FIT,WORST_FIT,STD_DEV';
    fs.writeFileSync(fd, init_buf + '\n' + buffer);
    fs.closeSync(fd);
}