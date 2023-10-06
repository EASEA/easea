package org.Graphers;

import org.Data.FitnessCoordinate;
import org.Gui.Frame;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.font.FontRenderContext;
import java.awt.font.LineMetrics;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;
import javax.swing.JPanel;

public class EvolutionGrapher extends JPanel {
    final int PAD = 30;
    private Frame frame;
    private Graphics2D g2;
    private int nbTicksAbcisse = 10;
    private int nbTicksOrdonnee = 5;
    private int ordinateLength = 0;
    private int absisseLength = 0;
    private int abcisseLowerBoundary = 0;
    private int abcisseUpperBoundary = 0;
    double xInc = 0.0;
    double scale = 0.0;
    private int nbMaxEval = 1;

    public EvolutionGrapher(Frame f) {
        this.frame = f;
    }

    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        this.g2 = (Graphics2D)g;
        this.g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        this.setBackground(Color.white);
        int w = this.getWidth();
        int h = this.getHeight();
        this.ordinateLength = h - 30 - 30;
        this.absisseLength = w - 30 - 30;
        this.abcisseLowerBoundary = 30;
        this.abcisseUpperBoundary = w - 30;
        int maxValue = (int)this.frame.getConsole().getData().getMaximumFitness(this.frame.getBestFitnessSelected(), this.frame.getAverageFitnessSelected(), this.frame.getStdDevSelected());
        if (maxValue % 5 == 0) {
            maxValue += 5;
        } else {
            maxValue += 5 - maxValue % 5;
        }

        this.g2.draw(new Line2D.Double(30.0, 30.0, 30.0, (double)(h - 30)));
        this.g2.draw(new Line2D.Double(30.0, (double)(h - 30), (double)(w - 30), (double)(h - 30)));
        Font font = this.g2.getFont();
        FontRenderContext frc = this.g2.getFontRenderContext();
        LineMetrics lm = font.getLineMetrics("0", frc);
        float sh = lm.getAscent() + lm.getDescent();
        float oldSize = (float)font.getSize();
        String legend = "Std Dev";
        this.g2.setPaint(Color.black);
        this.g2.setFont(this.g2.getFont().deriveFont(10.0F));
        frc = this.g2.getFontRenderContext();
        lm = font.getLineMetrics("0", frc);
        float stringWidth = (float)font.getStringBounds(legend, frc).getWidth();
        float stringHeight = (float)font.getStringBounds(legend, frc).getHeight();
        this.g2.drawString(legend, (int)((float)w - stringWidth), (int)stringHeight);
        this.g2.setPaint(Color.blue);
        this.g2.draw(new Line2D.Double((double)((float)w - stringWidth - 22.0F), (double)stringHeight * 0.75, (double)((float)w - stringWidth - 2.0F), (double)stringHeight * 0.75));
        float previousX = (float)w - stringWidth - 22.0F - 2.0F;
        legend = "Avg Fitness";
        this.g2.setPaint(Color.black);
        stringWidth = (float)font.getStringBounds(legend, frc).getWidth();
        stringHeight = (float)font.getStringBounds(legend, frc).getHeight();
        this.g2.drawString(legend, (int)(previousX - stringWidth), (int)stringHeight);
        this.g2.setPaint(Color.red);
        this.g2.draw(new Line2D.Double((double)(previousX - stringWidth - 22.0F), (double)stringHeight * 0.75, (double)(previousX - stringWidth - 2.0F), (double)stringHeight * 0.75));
        previousX = previousX - stringWidth - 22.0F - 2.0F;
        legend = "Best Fitness";
        this.g2.setPaint(Color.black);
        stringWidth = (float)font.getStringBounds(legend, frc).getWidth();
        stringHeight = (float)font.getStringBounds(legend, frc).getHeight();
        this.g2.drawString(legend, (int)(previousX - stringWidth), (int)stringHeight);
        this.g2.setPaint(Color.green);
        this.g2.draw(new Line2D.Double((double)(previousX - stringWidth - 22.0F), (double)stringHeight * 0.75, (double)(previousX - stringWidth - 2.0F), (double)stringHeight * 0.75));
        this.g2.setPaint(Color.black);
        String s = "Fitness";
        float sy = 15.0F;
        float sw = (float)font.getStringBounds(s, frc).getWidth();
        float sx = 15.0F;
        this.g2.drawString(s, sx, sy);
        s = "Nb of evaluations";
        sy = (float)(h - 30) + (30.0F - sh) / 2.0F + lm.getAscent() + 6.0F;
        sw = (float)font.getStringBounds(s, frc).getWidth();
        sx = ((float)w - sw) / 2.0F;
        this.g2.drawString(s, sx, sy);

        int i;
        int tick;
        for(i = 0; i < this.nbTicksAbcisse; ++i) {
            tick = this.absisseLength / this.nbTicksAbcisse;
            this.g2.draw(new Line2D.Double((double)(30 + (i + 1) * tick), (double)(h - 30), (double)(30 + (i + 1) * tick), (double)(h - 30 - 2)));
        }

        for(i = 0; i < this.nbTicksOrdonnee; ++i) {
            tick = this.ordinateLength / this.nbTicksOrdonnee;
            this.g2.draw(new Line2D.Double(30.0, (double)(h - 30 - (i + 1) * tick), 32.0, (double)(h - 30 - (i + 1) * tick)));
        }

        s = "0";
        sx = 23.0F;
        sy = (float)(h - 30) + lm.getAscent();
        this.g2.drawString(s, sx, sy);

        float f1;
        for(i = 1; i <= this.nbTicksAbcisse; ++i) {
            tick = this.absisseLength / this.nbTicksAbcisse;
            f1 = (float)this.nbMaxEval / (float)this.nbTicksAbcisse;
            f1 *= (float)i;
            if (Math.floor((double)f1) == (double)f1) {
                s = "" + (int)f1;
            } else {
                f1 = (float)Math.round(f1 * 100.0F) / 100.0F;
                s = "" + f1;
            }

            sy = (float)(h - 30) + lm.getAscent();
            sw = (float)font.getStringBounds(s, frc).getWidth();
            sx = (float)(30 + i * tick) - sw / 2.0F;
            this.g2.drawString(s, sx, sy);
        }

        for(i = 1; i <= this.nbTicksOrdonnee; ++i) {
            tick = this.ordinateLength / this.nbTicksOrdonnee;
            f1 = (float)maxValue / (float)this.nbTicksOrdonnee;
            f1 *= (float)i;
            if (Math.floor((double)f1) == (double)f1) {
                s = "" + (int)f1;
            } else {
                f1 = (float)Math.round(f1 * 100.0F) / 100.0F;
                s = "" + f1;
            }

            s = s + " ";
            sw = (float)font.getStringBounds(s, frc).getWidth();
            this.g2.drawString(s, 30.0F - sw, (float)(h - 30 - i * tick));
        }

        if (this.frame.getWithGridSelected()) {
            this.g2.setPaint(Color.gray.brighter());

            for(i = 0; i < this.nbTicksAbcisse; ++i) {
                tick = this.absisseLength / this.nbTicksAbcisse;
                this.g2.draw(new Line2D.Double((double)(30 + (i + 1) * tick), 30.0, (double)(30 + (i + 1) * tick), (double)(h - 30)));
            }

            for(i = 0; i < this.nbTicksOrdonnee; ++i) {
                tick = this.ordinateLength / this.nbTicksOrdonnee;
                this.g2.draw(new Line2D.Double(30.0, (double)(h - 30 - (i + 1) * tick), (double)(w - 30), (double)(h - 30 - (i + 1) * tick)));
            }
        }

        this.xInc = (double)(w - 60) / (double)this.nbMaxEval;
        this.scale = (double)(h - 60) / (double)maxValue;
        double y;
        double x2;
        double y2;
        double x;
        if (this.frame.getBestFitnessSelected()) {
            this.g2.setPaint(Color.green);

            for(i = 0; i < this.frame.getConsole().getData().getBestFitnessList().size() - 1; ++i) {
                x = 30.0 + (double)((FitnessCoordinate)this.frame.getConsole().getData().getBestFitnessList().get(i)).getNbEval() * this.xInc;
                y = (double)(h - 30) - this.scale * ((FitnessCoordinate)this.frame.getConsole().getData().getBestFitnessList().get(i)).getValue();
                x2 = 30.0 + (double)((FitnessCoordinate)this.frame.getConsole().getData().getBestFitnessList().get(i + 1)).getNbEval() * this.xInc;
                y2 = (double)(h - 30) - this.scale * ((FitnessCoordinate)this.frame.getConsole().getData().getBestFitnessList().get(i + 1)).getValue();
                this.g2.draw(new Line2D.Double(x, y, x2, y2));
            }

            if (this.frame.getWithPointsSelected()) {
                for(i = 0; i < this.frame.getConsole().getData().getBestFitnessList().size(); ++i) {
                    x = 30.0 + (double)((FitnessCoordinate)this.frame.getConsole().getData().getBestFitnessList().get(i)).getNbEval() * this.xInc;
                    y = (double)(h - 30) - ((FitnessCoordinate)this.frame.getConsole().getData().getBestFitnessList().get(i)).getValue() * this.scale;
                    if (((FitnessCoordinate)this.frame.getConsole().getData().getBestFitnessList().get(i)).getIsImmigrant()) {
                        this.g2.setPaint(Color.orange);
                    } else {
                        this.g2.setPaint(Color.green.darker());
                    }

                    this.g2.fill(new Ellipse2D.Double(x - 2.0, y - 2.0, 4.0, 4.0));
                    ((FitnessCoordinate)this.frame.getConsole().getData().getBestFitnessList().get(i)).setX(x);
                    ((FitnessCoordinate)this.frame.getConsole().getData().getBestFitnessList().get(i)).setY(y);
                }
            }
        }

        if (this.frame.getAverageFitnessSelected()) {
            this.g2.setPaint(Color.red);

            for(i = 0; i < this.frame.getConsole().getData().getAverageFitnessList().size() - 1; ++i) {
                x = 30.0 + (double)((FitnessCoordinate)this.frame.getConsole().getData().getAverageFitnessList().get(i)).getNbEval() * this.xInc;
                y = (double)(h - 30) - this.scale * ((FitnessCoordinate)this.frame.getConsole().getData().getAverageFitnessList().get(i)).getValue();
                x2 = 30.0 + (double)((FitnessCoordinate)this.frame.getConsole().getData().getAverageFitnessList().get(i + 1)).getNbEval() * this.xInc;
                y2 = (double)(h - 30) - this.scale * ((FitnessCoordinate)this.frame.getConsole().getData().getAverageFitnessList().get(i + 1)).getValue();
                this.g2.draw(new Line2D.Double(x, y, x2, y2));
            }

            if (this.frame.getWithPointsSelected()) {
                this.g2.setPaint(Color.red.darker());

                for(i = 0; i < this.frame.getConsole().getData().getAverageFitnessList().size(); ++i) {
                    x = 30.0 + (double)((FitnessCoordinate)this.frame.getConsole().getData().getAverageFitnessList().get(i)).getNbEval() * this.xInc;
                    y = (double)(h - 30) - ((FitnessCoordinate)this.frame.getConsole().getData().getAverageFitnessList().get(i)).getValue() * this.scale;
                    this.g2.fill(new Ellipse2D.Double(x - 2.0, y - 2.0, 4.0, 4.0));
                    ((FitnessCoordinate)this.frame.getConsole().getData().getAverageFitnessList().get(i)).setX(x);
                    ((FitnessCoordinate)this.frame.getConsole().getData().getAverageFitnessList().get(i)).setY(y);
                }
            }
        }

        if (this.frame.getStdDevSelected()) {
            this.g2.setPaint(Color.blue);

            for(i = 0; i < this.frame.getConsole().getData().getStdDevList().size() - 1; ++i) {
                x = 30.0 + (double)((FitnessCoordinate)this.frame.getConsole().getData().getStdDevList().get(i)).getNbEval() * this.xInc;
                y = (double)(h - 30) - this.scale * ((FitnessCoordinate)this.frame.getConsole().getData().getStdDevList().get(i)).getValue();
                x2 = 30.0 + (double)((FitnessCoordinate)this.frame.getConsole().getData().getStdDevList().get(i + 1)).getNbEval() * this.xInc;
                y2 = (double)(h - 30) - this.scale * ((FitnessCoordinate)this.frame.getConsole().getData().getStdDevList().get(i + 1)).getValue();
                this.g2.draw(new Line2D.Double(x, y, x2, y2));
            }

            if (this.frame.getWithPointsSelected()) {
                this.g2.setPaint(Color.blue.darker());

                for(i = 0; i < this.frame.getConsole().getData().getStdDevList().size(); ++i) {
                    x = 30.0 + (double)((FitnessCoordinate)this.frame.getConsole().getData().getStdDevList().get(i)).getNbEval() * this.xInc;
                    y = (double)(h - 30) - ((FitnessCoordinate)this.frame.getConsole().getData().getStdDevList().get(i)).getValue() * this.scale;
                    this.g2.fill(new Ellipse2D.Double(x - 2.0, y - 2.0, 4.0, 4.0));
                    ((FitnessCoordinate)this.frame.getConsole().getData().getStdDevList().get(i)).setX(x);
                    ((FitnessCoordinate)this.frame.getConsole().getData().getStdDevList().get(i)).setY(y);
                }
            }
        }

    }

    public int getAbcisseLowerBoundary() {
        return this.abcisseLowerBoundary;
    }

    public int getAbcisseUpperBoundary() {
        return this.abcisseUpperBoundary;
    }

    public Graphics2D getGraph() {
        return this.g2;
    }

    public void setNbMaxEval(int nbMaxEval) {
        this.nbMaxEval = nbMaxEval;
    }
}
