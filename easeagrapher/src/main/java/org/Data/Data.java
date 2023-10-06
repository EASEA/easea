package org.Data;

import java.util.ArrayList;

public class Data {
    private ArrayList<FitnessCoordinate> bestFitness = new ArrayList();
    private ArrayList<FitnessCoordinate> averageFitness = new ArrayList();
    private ArrayList<FitnessCoordinate> stdDev = new ArrayList();
    private ArrayList<IslandStatCoordinate> numberOfImmigrants = new ArrayList();
    private ArrayList<IslandStatCoordinate> numberOfImmgrantReproductions = new ArrayList();

    public Data() {
    }

    public ArrayList<FitnessCoordinate> getBestFitnessList() {
        return this.bestFitness;
    }

    public ArrayList<FitnessCoordinate> getAverageFitnessList() {
        return this.averageFitness;
    }

    public ArrayList<FitnessCoordinate> getStdDevList() {
        return this.stdDev;
    }

    public ArrayList<IslandStatCoordinate> getNumberOfImmigrantList() {
        return this.numberOfImmigrants;
    }

    public ArrayList<IslandStatCoordinate> getNumberOfImmigrantReproductionList() {
        return this.numberOfImmgrantReproductions;
    }

    public void addBestFitness(int nbEval, double value) {
        this.bestFitness.add(new FitnessCoordinate(nbEval, value));
    }

    public void addAverageFitness(int nbEval, double value) {
        this.averageFitness.add(new FitnessCoordinate(nbEval, value));
    }

    public void addStdDev(int nbEval, double value) {
        this.stdDev.add(new FitnessCoordinate(nbEval, value));
    }

    public void addImmigrantNumber(int nbGen, double value) {
        this.numberOfImmigrants.add(new IslandStatCoordinate(nbGen, value));
    }

    public void addImmigrantReproductionNumber(int nbGen, double value) {
        this.numberOfImmgrantReproductions.add(new IslandStatCoordinate(nbGen, value));
    }

    public double getMaximumFitness(boolean bestFitnessSelected, boolean averageFitnessSelected, boolean stdDevSelected) {
        double max = -1.7976931348623157E308;

        for(int i = 0; i < this.bestFitness.size(); ++i) {
            if (bestFitnessSelected && ((FitnessCoordinate)this.bestFitness.get(i)).getValue() > max) {
                max = ((FitnessCoordinate)this.bestFitness.get(i)).getValue();
            }

            if (averageFitnessSelected && ((FitnessCoordinate)this.averageFitness.get(i)).getValue() > max) {
                max = ((FitnessCoordinate)this.averageFitness.get(i)).getValue();
            }

            if (stdDevSelected && ((FitnessCoordinate)this.stdDev.get(i)).getValue() > max) {
                max = ((FitnessCoordinate)this.stdDev.get(i)).getValue();
            }
        }

        return max;
    }

    public double getMaximumNumberImmigrantValue() {
        double max = -1.7976931348623157E308;

        for(int i = 0; i < this.numberOfImmigrants.size(); ++i) {
            if (((IslandStatCoordinate)this.numberOfImmigrants.get(i)).getValue() > max) {
                max = ((IslandStatCoordinate)this.numberOfImmigrants.get(i)).getValue();
            }
        }

        return max;
    }

    public double getMaximumNumberImmigrantReproductionValue() {
        double max = -1.7976931348623157E308;

        for(int i = 0; i < this.numberOfImmigrants.size(); ++i) {
            if (((IslandStatCoordinate)this.numberOfImmgrantReproductions.get(i)).getValue() > max) {
                max = ((IslandStatCoordinate)this.numberOfImmgrantReproductions.get(i)).getValue();
            }
        }

        return max;
    }
}