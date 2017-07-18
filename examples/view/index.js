var garden = angular.module('garden' , []);


garden.controller('gardenController', function ($scope,$http,$timeout,$interval) {

    $scope.data = {
    	'images' : [],
        'training_sessions' : 0
    }

    $scope.randomizeImage = function() {
        $scope.data.images = []
        for (index=0 ; index < 10 ; index++ ) {
            $scope.data.images.push( Math.floor( Math.random() * 10000 ) )
        }
    }

    $scope.learn = function() {
        console.log("LEARNING...");
        $http({ method : "GET" , url : "/learn" , cache: false}).then(function successCallback(result) {
            console.log("...DONE");
            $scope.data.training_sessions += 1;
            $scope.randomizeImage();
            if ($scope.data.training_sessions < 40) {
                setTimeout( $scope.learn , 2000 );
            }
        })
    }

});