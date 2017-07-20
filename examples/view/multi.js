var garden = angular.module('garden' , []);


garden.controller('gardenController', function ($scope,$http,$timeout,$interval) {

    $scope.data = {
    	'images' : [],
        'training_sessions' : [0,0,0,0,0,0,0],
        'similar_images' : [],
        'difference_images' : [],
        'iterations' : 1,
        'prev' : 1
    }

    $scope.randomizeImage = function() {
        $scope.data.images = []
        for (index=0 ; index < 10 ; index++ ) {
            $scope.data.images.push( Math.floor( Math.random() * 10000 ) )
        }
    }

    $scope.reset_session = function() {
        $http({method:"GET" , url : "/reset_session" , cache: false}).then(function successCallback(result) {
            console.log("resetSession");
            $scope.randomizeImage();
        })
    }

    $scope.update_embeddings = function() {
        $http({method:"GET" , url : "/update_embeddings" , cache: false}).then(function successCallback(result) {
            console.log("update_embeddings");
        })
    }

    $scope.similar = function(index) {
        console.log("calling similar ...");
        $http({method:"GET" , url : "/similar/"+index , cache: false}).then(function successCallback(result) {
            console.log("... done");
            $scope.data.similar_images = result.data.response
        })
    }

    $scope.subtract = function(index) {
        console.log("calling subtract ...");
        $http({method:"GET" , url : "/difference/"+$scope.data.similar_images[0]+"/"+index , cache: false}).then(function successCallback(result) {
            console.log("... done");
            $scope.data.difference_images = result.data.response
        })
    }

    $scope.learn = function(index) {
        console.log("LEARNING...");
        $http({ method : "GET" , url : "/learn/"+index , cache: false}).then(function successCallback(result) {
            console.log("...DONE");
            $scope.data.training_sessions[index] += 1;
            $scope.randomizeImage();

            if ($scope.data.training_sessions[index] % (1 * $scope.data.iterations) != 0) {
                setTimeout( function() { $scope.learn(index) } , 5000 );
            }
        })
    }

});