package org.Gui;

import org.Data.FitnessCoordinate;
import org.Graphers.EvolutionGrapher;
import org.Graphers.ImmigrantNumberStatsGrapher;
import org.Graphers.ImmigrantReproductionNumberStatsGrapher;
import Main.Console;
import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Graphics;
import java.awt.GraphicsConfiguration;
import java.awt.GraphicsDevice;
import java.awt.GraphicsEnvironment;
import java.awt.HeadlessException;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.ImageObserver;
import java.awt.image.PixelGrabber;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JSeparator;
import javax.swing.JSplitPane;
import javax.swing.ToolTipManager;

public class Frame extends JFrame implements ActionListener, MouseMotionListener {
    private Console console;
    private EvolutionGrapher evolutionGraph;
    private ImmigrantNumberStatsGrapher immigrantNumberStatsGraph;
    private ImmigrantReproductionNumberStatsGrapher immigrantReproductionNumberStatsGrapher;
    private JPanel middlePanel;
    private JPanel evolutionPanel;
    private JPanel islandImmigrantNumberPanel;
    private JPanel islandImmigrantReproductionNumberPanel;
    private JPanel leftPanel;
    private JCheckBox bestFitnessCheckBox;
    private JCheckBox averageFitnessCheckBox;
    private JCheckBox stdDevCheckBox;
    private JCheckBox withPointCheckBox;
    private JCheckBox withGridCheckBox;
    private JSplitPane islandModelSplitPane;
    private JSplitPane islandModelStatsSplitPane;
    private JButton saveFileButton;

    public Frame(Console console) {
        this.console = console;
        this.setDefaultCloseOperation(3);
        this.setLayout(new BorderLayout());
        this.evolutionGraph = new EvolutionGrapher(this);
        this.middlePanel = new JPanel();
        this.middlePanel.setLayout(new BoxLayout(this.middlePanel, 3));
        this.middlePanel.addMouseMotionListener(this);
        this.add(this.middlePanel, "Center");
        this.evolutionPanel = new JPanel();
        this.evolutionPanel.setLayout(new BorderLayout());
        this.evolutionPanel.addMouseMotionListener(this);
        this.middlePanel.add(this.evolutionPanel);
        this.leftPanel = new JPanel();
        this.leftPanel.setLayout(new BoxLayout(this.leftPanel, 3));
        this.add(this.leftPanel, "East");
        this.bestFitnessCheckBox = new JCheckBox("Best Fitness");
        this.bestFitnessCheckBox.setSelected(true);
        this.bestFitnessCheckBox.addActionListener(this);
        this.leftPanel.add(this.bestFitnessCheckBox);
        this.averageFitnessCheckBox = new JCheckBox("Avg Fitness");
        this.averageFitnessCheckBox.setSelected(true);
        this.averageFitnessCheckBox.addActionListener(this);
        this.leftPanel.add(this.averageFitnessCheckBox);
        this.stdDevCheckBox = new JCheckBox("Std Dev");
        this.stdDevCheckBox.setSelected(true);
        this.stdDevCheckBox.addActionListener(this);
        this.leftPanel.add(this.stdDevCheckBox);
        this.leftPanel.add(new JSeparator(0));
        this.withPointCheckBox = new JCheckBox("With point");
        this.withPointCheckBox.setSelected(true);
        this.withPointCheckBox.addActionListener(this);
        this.leftPanel.add(this.withPointCheckBox);
        this.withGridCheckBox = new JCheckBox("With grid");
        this.withGridCheckBox.setSelected(true);
        this.withGridCheckBox.addActionListener(this);
        this.leftPanel.add(this.withGridCheckBox);
        this.saveFileButton = new JButton("Save Graph");
        this.saveFileButton.addActionListener(this);
        this.leftPanel.add(this.saveFileButton);
    }

    public void addGraph() {
        this.evolutionPanel.add(this.evolutionGraph, "Center");
        if (this.console.isIslandModel()) {
            this.islandImmigrantNumberPanel.add(this.immigrantNumberStatsGraph, "Center");
            this.islandImmigrantReproductionNumberPanel.add(this.immigrantReproductionNumberStatsGrapher, "Center");
        }

    }

    public void setIslandModel() {
        this.islandImmigrantNumberPanel = new JPanel();
        this.islandImmigrantNumberPanel.setLayout(new BorderLayout());
        this.islandImmigrantReproductionNumberPanel = new JPanel();
        this.islandImmigrantReproductionNumberPanel.setLayout(new BorderLayout());
        this.immigrantNumberStatsGraph = new ImmigrantNumberStatsGrapher(this);
        this.immigrantReproductionNumberStatsGrapher = new ImmigrantReproductionNumberStatsGrapher(this);
        this.islandModelStatsSplitPane = new JSplitPane(1);
        this.islandModelStatsSplitPane.setResizeWeight(0.5);
        this.islandModelStatsSplitPane.setLeftComponent(this.islandImmigrantNumberPanel);
        this.islandModelStatsSplitPane.setRightComponent(this.islandImmigrantReproductionNumberPanel);
        this.islandModelSplitPane = new JSplitPane(0);
        this.islandModelSplitPane.setLeftComponent(this.evolutionPanel);
        this.islandModelSplitPane.setRightComponent(this.islandModelStatsSplitPane);
        this.islandModelSplitPane.setOneTouchExpandable(true);
        this.islandModelSplitPane.setResizeWeight(0.65);
        this.remove(this.middlePanel);
        this.validate();
        this.add(this.islandModelSplitPane, "Center");
        this.validate();
    }

    public boolean getBestFitnessSelected() {
        return this.bestFitnessCheckBox.isSelected();
    }

    public boolean getAverageFitnessSelected() {
        return this.averageFitnessCheckBox.isSelected();
    }

    public boolean getStdDevSelected() {
        return this.stdDevCheckBox.isSelected();
    }

    public boolean getWithPointsSelected() {
        return this.withPointCheckBox.isSelected();
    }

    public boolean getWithGridSelected() {
        return this.withGridCheckBox.isSelected();
    }

    public Console getConsole() {
        return this.console;
    }

    public EvolutionGrapher getEvolutionGraph() {
        return this.evolutionGraph;
    }

    public ImmigrantNumberStatsGrapher getImmigrantNumberStatGraph() {
        return this.immigrantNumberStatsGraph;
    }

    public ImmigrantReproductionNumberStatsGrapher getImmigrantReproductionNumberStatGraph() {
        return this.immigrantReproductionNumberStatsGrapher;
    }

    private BufferedImage toBufferedImage(Image image) {
        if (image instanceof BufferedImage) {
            return (BufferedImage)image;
        } else {
            image = (new ImageIcon(image)).getImage();
            boolean hasAlpha = hasAlpha(image);
            BufferedImage bimage = null;
            GraphicsEnvironment ge = GraphicsEnvironment.getLocalGraphicsEnvironment();

            byte type;
            try {
                type = 1;
                if (hasAlpha) {
                    type = 2;
                }

                GraphicsDevice gs = ge.getDefaultScreenDevice();
                GraphicsConfiguration gc = gs.getDefaultConfiguration();
                bimage = gc.createCompatibleImage(image.getWidth((ImageObserver)null), image.getHeight((ImageObserver)null), type);
            } catch (HeadlessException var8) {
            }

            if (bimage == null) {
                type = 1;
                if (hasAlpha) {
                    type = 2;
                }

                bimage = new BufferedImage(image.getWidth((ImageObserver)null), image.getHeight((ImageObserver)null), type);
            }

            Graphics g = bimage.createGraphics();
            g.drawImage(image, 0, 0, image.getWidth((ImageObserver)null), image.getHeight((ImageObserver)null), (ImageObserver)null);
            g.dispose();
            return bimage;
        }
    }

    public static boolean hasAlpha(Image image) {
        if (image instanceof BufferedImage) {
            BufferedImage bimage = (BufferedImage)image;
            return bimage.getColorModel().hasAlpha();
        } else {
            PixelGrabber pg = new PixelGrabber(image, 0, 0, 1, 1, false);

            try {
                pg.grabPixels();
            } catch (InterruptedException var3) {
            }

            ColorModel cm = pg.getColorModel();
            return cm.hasAlpha();
        }
    }

    public void actionPerformed(ActionEvent arg0) {
        if (arg0.getSource() == this.saveFileButton) {
            try {
                Image img = this.createImage(this.evolutionPanel.getWidth(), this.evolutionPanel.getHeight());
                Graphics g = img.getGraphics();
                this.paint(g);
                File outputfile = new File("saved.png");
                ImageIO.write(this.toBufferedImage(img), "png", outputfile);
                JOptionPane.showMessageDialog((Component)null, "Image saved to " + outputfile.toString());
                g.dispose();
            } catch (IOException var5) {
            }
        } else {
            this.repaint();
        }

    }

    public void mouseDragged(MouseEvent arg0) {
    }

    public void mouseMoved(MouseEvent arg0) {
        if (arg0.getX() >= this.evolutionGraph.getAbcisseLowerBoundary() && arg0.getX() <= this.evolutionGraph.getAbcisseUpperBoundary()) {
            int iFound = -1;

            for(int i = 0; i < this.console.getData().getBestFitnessList().size(); ++i) {
                if ((double)arg0.getX() >= ((FitnessCoordinate)this.console.getData().getBestFitnessList().get(i)).getX() - 2.0 && (double)arg0.getX() <= ((FitnessCoordinate)this.console.getData().getBestFitnessList().get(i)).getX() + 2.0) {
                    iFound = i;
                    i = this.console.getData().getBestFitnessList().size();
                }
            }

            if (iFound >= 0) {
                String tooltip;
                if ((double)arg0.getY() >= ((FitnessCoordinate)this.console.getData().getBestFitnessList().get(iFound)).getY() - 2.0 && (double)arg0.getY() <= ((FitnessCoordinate)this.console.getData().getBestFitnessList().get(iFound)).getY() + 2.0 && this.getBestFitnessSelected() && this.getWithPointsSelected()) {
                    tooltip = "";
                    if (((FitnessCoordinate)this.console.getData().getBestFitnessList().get(iFound)).getIsImmigrant()) {
                        tooltip = tooltip + "Immigrant ";
                    }

                    tooltip = tooltip + "Nb Evaluation: " + ((FitnessCoordinate)this.console.getData().getBestFitnessList().get(iFound)).getNbEval() + "\n";
                    tooltip = tooltip + "Fitness value: " + ((FitnessCoordinate)this.console.getData().getBestFitnessList().get(iFound)).getValue();
                    this.evolutionPanel.setToolTipText(tooltip);
                    ToolTipManager.sharedInstance().mouseMoved(new MouseEvent(this.evolutionPanel, 0, 0L, 0, arg0.getX(), arg0.getY(), 0, false));
                    return;
                }

                if ((double)arg0.getY() >= ((FitnessCoordinate)this.console.getData().getAverageFitnessList().get(iFound)).getY() - 2.0 && (double)arg0.getY() <= ((FitnessCoordinate)this.console.getData().getAverageFitnessList().get(iFound)).getY() + 2.0 && this.getAverageFitnessSelected() && this.getWithPointsSelected()) {
                    tooltip = "Nb Evaluation: " + ((FitnessCoordinate)this.console.getData().getAverageFitnessList().get(iFound)).getNbEval() + "\n";
                    tooltip = tooltip + "Average value: " + ((FitnessCoordinate)this.console.getData().getAverageFitnessList().get(iFound)).getValue();
                    this.evolutionPanel.setToolTipText(tooltip);
                    ToolTipManager.sharedInstance().mouseMoved(new MouseEvent(this.evolutionPanel, 0, 0L, 0, arg0.getX(), arg0.getY(), 0, false));
                    return;
                }

                if ((double)arg0.getY() >= ((FitnessCoordinate)this.console.getData().getStdDevList().get(iFound)).getY() - 2.0 && (double)arg0.getY() <= ((FitnessCoordinate)this.console.getData().getStdDevList().get(iFound)).getY() + 2.0 && this.getStdDevSelected() && this.getWithPointsSelected()) {
                    tooltip = "Nb Evaluation: " + ((FitnessCoordinate)this.console.getData().getStdDevList().get(iFound)).getNbEval() + "\n";
                    tooltip = tooltip + "Std Dev value: " + ((FitnessCoordinate)this.console.getData().getStdDevList().get(iFound)).getValue();
                    this.evolutionPanel.setToolTipText(tooltip);
                    ToolTipManager.sharedInstance().mouseMoved(new MouseEvent(this.evolutionPanel, 0, 0L, 0, arg0.getX(), arg0.getY(), 0, false));
                    return;
                }
            }
        }

        this.evolutionPanel.setToolTipText((String)null);
        this.repaint();
    }
}
