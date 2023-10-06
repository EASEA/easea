package org.Data;

public class IslandStatCoordinate {
    private int nbGen;
    private double value;
    private double x;
    private double y;

    public IslandStatCoordinate(int nbGen, double value) {
        this.nbGen = nbGen;
        this.value = value;
    }

    public int getNbGen() {
        return this.nbGen;
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
}