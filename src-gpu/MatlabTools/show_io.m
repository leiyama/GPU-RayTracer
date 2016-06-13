figure;
axis equal;
hold on;

file = fopen('iotest_results1.txt');
C = fscanf(file, '%f %f %f %f', [3 Inf]);
C = C';

X = C(:, 1);
Y = C(:, 2);
Z = C(:, 3);

for i=1:length(X)
    plot3(X(i), Y(i), Z(i), '.b');
end

file = fopen('iotest_results2.txt');
C = fscanf(file, '%f %f %f %f', [3 Inf]);
C = C';

X = C(:, 1);
Y = C(:, 2);
Z = C(:, 3);

for i=1:length(X)
    plot3(X(i), Y(i), Z(i), '.r');
end

file = fopen('iotest_results3.txt');
C = fscanf(file, '%f %f %f %f', [3 Inf]);
C = C';

X = C(:, 1);
Y = C(:, 2);
Z = C(:, 3);

for i=1:length(X)
    plot3(X(i), Y(i), Z(i), '.g');
end