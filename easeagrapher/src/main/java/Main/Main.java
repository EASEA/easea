package Main;

public class Main {
    public static void main(String[] args) {
        final Console c = new Console();
        Thread t = new Thread(new Runnable() {
            public void run() {
                c.waitForCommand();
            }
        });
        t.start();
    }
}
