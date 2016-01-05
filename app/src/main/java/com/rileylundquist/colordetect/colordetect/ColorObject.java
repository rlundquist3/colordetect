package com.rileylundquist.colordetect.colordetect;

import org.opencv.core.Scalar;

/**
 * Created by rlundquist3 on 12/11/15.
 */
public class ColorObject {

    private int xPos;
    private int yPos;
    private String type;
    private Scalar hsvMin;
    private Scalar hsvMax;
    private Scalar color;

    public ColorObject() {
        setType("Object");
        setColor(new Scalar(0, 0, 0));
    }

    public ColorObject(String color) {
        setType(color);

        if (color.equals("blue")) {
            setHsvMin(new Scalar(92, 0, 0));
            setHsvMax(new Scalar(124, 256, 256));
            setColor(new Scalar(255, 0, 0));
        } else if (color.equals("green")) {
            setHsvMin(new Scalar(34, 50, 50));
            setHsvMax(new Scalar(80, 220, 200));
            setColor(new Scalar(0, 255, 0));
        } if (color.equals("yellow")) {
            setHsvMin(new Scalar(20, 124, 123));
            setHsvMax(new Scalar(30, 256, 256));
            setColor(new Scalar(0, 255, 255));
        } if (color.equals("red")) {
            setHsvMin(new Scalar(0, 200, 0));
            setHsvMax(new Scalar(19, 255, 255));
            setColor(new Scalar(0, 0, 255));
        }
    }

    public int getxPos() {
        return xPos;
    }

    public void setxPos(int xPos) {
        this.xPos = xPos;
    }

    public int getyPos() {
        return yPos;
    }

    public void setyPos(int yPos) {
        this.yPos = yPos;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public Scalar getHsvMin() {
        return hsvMin;
    }

    public void setHsvMin(Scalar hsvMin) {
        this.hsvMin = hsvMin;
    }

    public Scalar getHsvMax() {
        return hsvMax;
    }

    public void setHsvMax(Scalar hsvMax) {
        this.hsvMax = hsvMax;
    }

    public Scalar getColor() {
        return color;
    }

    public void setColor(Scalar color) {
        this.color = color;
    }
}
