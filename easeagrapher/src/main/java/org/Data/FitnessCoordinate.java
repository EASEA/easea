package org.Data;

public class FitnessCoordinate {
    private int nbEval;
    private double value;
    private double x;
    private double y;
    private boolean isImmigrant = false;

    public FitnessCoordinate(int nbEval, double value) {
        this.nbEval = nbEval;
        this.value = value;
    }

    public int getNbEval() {
        return this.nbEval;
    }

    public double getValue() {
        return this.value;
    }

    public double getX() {
        return this.x;
    }

    public double getY() {
        return this.y;
    }

    public void setX(double x) {
        this.x = x;
    }

    public void setY(double y) {
        this.y = y;
    }

    public void setIsImmigrant(boolean b) {
        this.isImmigrant = b;
    }

    public boolean getIsImmigrant() {
        return this.isImmigrant;
    }
}
