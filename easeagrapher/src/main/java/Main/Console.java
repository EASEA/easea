package Main;

import org.Data.Data;
import org.Data.FitnessCoordinate;
import org.Gui.Frame;
import java.util.Scanner;

public class Console {
    private boolean wait = true;
    private Frame frame = new Frame(this);
    private String title = "";
    private boolean remoteIslandModel = false;
    private Data data = new Data();

    public Console() {
    }

    public void waitForCommand() {
        Scanner in = new Scanner(System.in);
        System.out.println("Waiting for command :");

        while(this.wait) {
            String command = in.nextLine();
            if (command.equals("repaint")) {
                this.repaint();
            } else if (command.equals("paint")) {
                this.paint();
            } else {
                String t;
                int max;
                if (command.startsWith("set max eval:")) {
                    t = command.substring("set max eval:".length(), command.length());
                    max = Integer.parseInt(t);
                    if (max % 10 != 0) {
                        max += 10 - max % 10;
                    }

                    this.frame.getEvolutionGraph().setNbMaxEval(max);
                } else if (command.startsWith("add coordinate:")) {
                    t = command.substring("add coordinate:".length(), command.length());
                    this.addData(t);
                } else if (command.startsWith("add stat:")) {
                    t = command.substring("add stat:".length(), command.length());
                    this.addStats(t);
                } else if (command.startsWith("set title:")) {
                    this.title = command.substring("set title:".length(), command.length());
                    this.frame.setTitle(this.title);
                } else if (command.startsWith("set immigrant")) {
                    ((FitnessCoordinate)this.data.getBestFitnessList().get(this.data.getBestFitnessList().size() - 1)).setIsImmigrant(true);
                } else if (command.startsWith("set max generation:")) {
                    t = command.substring("set max generation:".length(), command.length());
                    max = Integer.parseInt(t);
                    if (max % 10 != 0) {
                        max += 10 - max % 10;
                    }

                    this.frame.getImmigrantNumberStatGraph().setNbMaxGen(max);
                    this.frame.getImmigrantReproductionNumberStatGraph().setNbMaxGen(max);
                } else if (command.startsWith("set island model")) {
                    this.remoteIslandModel = true;
                    this.frame.setIslandModel();
                }
            }
        }

        in.close();
    }

    private void addData(String t) {
        int nbEval = Integer.parseInt(t.substring(0, t.indexOf(";")));
        t = t.substring(t.indexOf(";") + 1, t.length());
        double bestFitness = Double.parseDouble(t.substring(0, t.indexOf(";")));
        t = t.substring(t.indexOf(";") + 1, t.length());
        double averageFitness = Double.parseDouble(t.substring(0, t.indexOf(";")));
        t = t.substring(t.indexOf(";") + 1, t.length());
        double stdDev = Double.parseDouble(t);
        this.data.addBestFitness(nbEval, bestFitness);
        this.data.addAverageFitness(nbEval, averageFitness);
        this.data.addStdDev(nbEval, stdDev);
    }

    private void addStats(String t) {
        int nbGen = Integer.parseInt(t.substring(0, t.indexOf(";")));
        t = t.substring(t.indexOf(";") + 1, t.length());
        double numberImmigrants = Double.parseDouble(t.substring(0, t.indexOf(";")));
        t = t.substring(t.indexOf(";") + 1, t.length());
        double numberImmigrantReproductions = Double.parseDouble(t);
        this.data.addImmigrantNumber(nbGen, numberImmigrants);
        this.data.addImmigrantReproductionNumber(nbGen, numberImmigrantReproductions);
    }

    private void repaint() {
        this.frame.repaint();
    }

    public void paint() {
        this.frame.addGraph();
        this.frame.setSize(800, 400);
        this.frame.setLocation(200, 200);
        this.frame.toFront();
        this.frame.setVisible(true);
    }

    public Data getData() {
        return this.data;
    }

    public boolean isIslandModel() {
        return this.remoteIslandModel;
    }
}